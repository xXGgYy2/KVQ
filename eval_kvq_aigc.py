#!/usr/bin/env python3
"""Evaluate KVQ checkpoints on an AIGC video list (filename MOS, whitespace separated)."""
import argparse, csv, inspect, math
from pathlib import Path
import numpy as np
import torch, yaml
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
import vqa.datasets as kvq_datasets
import vqa.models as kvq_models

SAMPLE_TYPES = ("resize", "fragments", "crop", "arp_resize", "arp_fragments")

def parse_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--video-root')
    p.add_argument('--mos-file')
    p.add_argument('--config', default='configs/test.yaml')
    p.add_argument('--dataset', help='Key under data:, e.g. test-aigc')
    p.add_argument('--checkpoint')
    p.add_argument('--output-dir')
    p.add_argument('--num-workers', type=int)
    p.add_argument('--device')
    p.add_argument('--no-amp', action='store_true')
    a=p.parse_args(argv)
    config=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    if not isinstance(config,dict): p.error('Config must be a YAML mapping')
    settings=config.get('evaluation',config.get('eval',{}))
    if not isinstance(settings,dict): p.error('evaluation must be a YAML mapping')
    datasets=config.get('data',{})
    if not isinstance(datasets,dict): p.error('data must be a YAML mapping')
    dataset_name=a.dataset or settings.get('dataset')
    if dataset_name is None:
        if 'test-aigc' in datasets:
            dataset_name='test-aigc'
        else:
            candidates=[key for key in datasets if key.startswith(('test-','val-'))]
            if len(candidates)==1: dataset_name=candidates[0]
    if dataset_name not in datasets:
        p.error(f'Select --dataset from {list(datasets)} or set evaluation.dataset in the config')
    a.dataset=dataset_name
    data_args=datasets[dataset_name].get('args',{})
    a.video_root=a.video_root or settings.get('video_root') or data_args.get('data_prefix')
    a.mos_file=a.mos_file or settings.get('mos_file') or data_args.get('anno_file')
    a.checkpoint=a.checkpoint or settings.get('checkpoint') or config.get('test_load_path')
    a.output_dir=a.output_dir or settings.get('output_dir') or f'results/{Path(a.config).stem}/{dataset_name}'
    a.num_workers=a.num_workers if a.num_workers is not None else settings.get('num_workers',0)
    a.device=a.device or settings.get('device','cuda:0')
    for name in ('video_root','mos_file','checkpoint'):
        if not getattr(a,name): p.error(f'--{name.replace("_","-")} is missing; set it in evaluation: or on the command line')
    return a,config

def read_rows(path, root):
    rows=[]
    for n,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip(): continue
        if ',' in line:
            fields=next(csv.reader([line],skipinitialspace=True))
            if len(fields)!=4: raise ValueError(f'{path}:{n}: expected name,-1,-1,mos')
            name,mos_text=fields[0].strip(),fields[3].strip()
        else:
            fields=line.rsplit(None,1)
            if len(fields)!=2: raise ValueError(f'{path}:{n}: expected filename MOS')
            name,mos_text=fields
        if not name: raise ValueError(f'{path}:{n}: empty filename')
        video=Path(name); video=(video if video.is_absolute() else Path(root)/video).resolve()
        mos=float(mos_text)
        if not math.isfinite(mos) or not video.is_file(): raise ValueError(f'{path}:{n}: invalid MOS or missing video: {video}')
        rows.append({'filename':str(video),'mos':mos})
    if len(rows)<2: raise ValueError('At least two labeled videos are required')
    if len({r['filename'] for r in rows})!=len(rows): raise ValueError('Duplicate video in MOS list')
    return rows

def make_dataset(config, rows, dataset_name):
    template=config['data'][dataset_name]
    options=dict(template['args']); options['phase']='test'; options['data_prefix']='/'
    annotation=Path('/tmp/kvq_aigc_labels.txt')
    annotation.write_text(''.join(f"{r['filename']}, -1, -1, {r['mos']:.12g}\n" for r in rows),encoding='utf-8')
    options['anno_file']=str(annotation)
    return getattr(kvq_datasets,template['type'])(options), options['sample_types']

def make_model(config, checkpoint, device):
    mc=config['model']; model=getattr(kvq_models,mc['type'])(**mc['args'])
    try: saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    except TypeError: saved=torch.load(checkpoint,map_location='cpu')
    state=saved.get('state_dict',saved) if isinstance(saved,dict) else saved
    if not isinstance(state,dict): raise TypeError(f'Unsupported checkpoint format: {checkpoint}')
    # DDP checkpoints may prefix every model key with "module.".
    state={k[7:] if k.startswith('module.') else k:v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()

def scalar(x):
    if torch.is_tensor(x): return float(x.detach().float().mean().cpu()) if x.numel() else math.nan
    return float(x) if isinstance(x,(float,int,np.number)) else math.nan

def outputs(raw):
    values=[]
    for x in (raw.values() if isinstance(raw,dict) else raw if isinstance(raw,(tuple,list)) else [raw]):
        v=scalar(x)
        if np.isfinite(v): values.append(v)
        if len(values)==4: break
    return values+[math.nan]*(4-len(values))

def call(model, inputs):
    sig=inspect.signature(model.forward); kw={k:v for k,v in {'cross_flag':False,'epoch':0,'inference':True}.items() if k in sig.parameters}
    return model(inputs,**kw)

def main():
    a,config=parse_args(); root=Path(a.video_root).resolve(); rows=read_rows(a.mos_file,root)
    device=torch.device(a.device)
    dataset,sample_types=make_dataset(config,rows,a.dataset)
    if len(dataset)!=len(rows): raise RuntimeError(f'Dataset resolved {len(dataset)} videos, expected {len(rows)}')
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=a.num_workers,pin_memory=device.type=='cuda')
    model=make_model(config,Path(a.checkpoint),device); done=[]
    for i,batch in enumerate(tqdm(loader,desc='AIGC')):
        vb=batch.get('video',batch); inputs={k:vb[k].to(device,non_blocking=True) for k in sample_types if k in vb and torch.is_tensor(vb[k])}
        with torch.inference_mode(), torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda' and not a.no_amp): raw=call(model,inputs)
        vals=outputs(raw); done.append({'filename':rows[i]['filename'],'mos':rows[i]['mos'],**{f'output{j}':vals[j] for j in range(4)}})
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); fields=['filename','mos','output0','output1','output2','output3']
    with (out/'per_video.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(done)
    metric=[]
    for j in range(4):
        p=np.array([r[f'output{j}'] for r in done]); y=np.array([r['mos'] for r in done]); ok=np.isfinite(p)
        metric.append({'branch':f'output{j}','n':int(ok.sum()),'PLCC':float(pearsonr(y[ok],p[ok]).statistic) if ok.sum()>1 and np.ptp(p[ok]) else math.nan,'SRCC':float(spearmanr(y[ok],p[ok]).statistic) if ok.sum()>1 and np.ptp(p[ok]) else math.nan,'KRCC':float(kendalltau(y[ok],p[ok]).statistic) if ok.sum()>1 and np.ptp(p[ok]) else math.nan})
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=['branch','n','PLCC','SRCC','KRCC']); w.writeheader(); w.writerows(metric)
    print(f'Results written to {out.resolve()}'); [print(m) for m in metric]
if __name__=='__main__': main()
