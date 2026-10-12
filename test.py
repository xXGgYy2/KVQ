import torch
import cv2
import random
import os.path as osp
import os
import matplotlib.pyplot as plt
import vqa.models as models
import vqa.datasets as datasets
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
import argparse
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

from time import time
from tqdm import tqdm
import pickle
import math

import wandb
import yaml

from functools import reduce
from thop import profile
import copy


def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        print(np.std(gt), np.mean(gt))
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]

def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device).unsqueeze(0)
    with torch.no_grad():
        flops, params = profile(model, (video, ))
    print(f"The FLOps of the Variant is {flops/1e9:.1f}G, with Params {params/1e6:.2f}M.")

def inference_set(inf_loader, model, device, best_, wandb_flag=False, ddp_flag=False, save_model=False, suffix='s', save_name="divide"):

    results = []
    best_s, best_p, best_k, best_r= best_
 
    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video, video_up = {}, {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                ## Reshape into clips
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
            if key + "_up" in data:
                video_up[key] = data[key+"_up"].to(device)
                ## Reshape into clips
                b, c, t, h, w = video_up[key].shape
                video_up[key] = video_up[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
            #.unsqueeze(0)
        with torch.no_grad():
            result["pr_labels"], feat, _, _, _= model(video)
            result["pr_labels"], feat = result["pr_labels"].cpu().numpy(), feat.cpu().numpy()
            if len(list(video_up.keys())) > 0:
                result["pr_labels_up"] = model(video_up).cpu().numpy()
                
        result["gt_label"] = data["gt_label"].item()
        del video, video_up
        # result['frame_inds'] = data['frame_inds']
        # del data
        results.append(result)
        
    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"][:]) for r in results]

    if ddp_flag:
        gt_labels = torch.tensor(gt_labels).to(device)
        pr_labels = torch.tensor(pr_labels).to(device)
        with torch.no_grad():
            all_gt_labels = [torch.zeros_like(gt_labels) for _ in range(torch.cuda.device_count())]
            torch.distributed.all_gather(all_gt_labels, gt_labels)
            all_pr_labels = [torch.zeros_like(pr_labels) for _ in range(torch.cuda.device_count())]
            torch.distributed.all_gather(all_pr_labels, pr_labels)
        gt_labels = torch.cat(all_gt_labels)
        pr_labels = torch.cat(all_pr_labels)
        gt_labels = gt_labels.tolist()
        pr_labels = pr_labels.tolist()
    pr_labels = rescale(pr_labels, gt_labels)
    print(len(pr_labels))


    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    if wandb_flag:
        if ddp_flag==False or (ddp_flag and int(os.environ["LOCAL_RANK"])==0): 
            wandb.log({f"val_{suffix}/SRCC-{suffix}": s, f"val_{suffix}/PLCC-{suffix}": p, f"val_{suffix}/KRCC-{suffix}": k, f"val_{suffix}/RMSE-{suffix}": r})
    del results, result #, video, video_up
    torch.cuda.empty_cache()

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    if wandb_flag:
        if ddp_flag==False or (ddp_flag and int(os.environ["LOCAL_RANK"])==0): 
            wandb.log(
                {
                    f"val_{suffix}/best_SRCC-{suffix}": best_s,
                    f"val_{suffix}/best_PLCC-{suffix}": best_p,
                    f"val_{suffix}/best_KRCC-{suffix}": best_k,
                    f"val_{suffix}/best_RMSE-{suffix}": best_r,
                }
            )

    if ddp_flag==False or (ddp_flag and int(os.environ["LOCAL_RANK"])==0): 
        print(
            f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
        )

    return best_s, best_p, best_k, best_r, s, p, k, r

    # torch.save(results, f'{args.save_dir}/results_{dataset.lower()}_s{32}*{32}_ens{args.famount}.pkl')


def init_seeds(seed=0):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="./configs/test.yaml", help="the config file"
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ## adaptively choose the device
    if config["ddp"] == True:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"


    if config['ddp']==False or (config['ddp'] and local_rank==0):
        print(config)
    ## defining model and loading checkpoint
    model = getattr(models, config["model"]["type"])(**config["model"]["args"]).to(device)


    val_datasets = {}
    for key in config["data"]:
        if key.startswith("val"):
            val_datasets[key] = getattr(datasets, 
                                        config["data"][key]["type"])(config["data"][key]["args"])
    val_loaders = {}
    valid_sampler = {}
    for key, val_dataset in val_datasets.items():
        if config['ddp']:
            valid_sampler[key] = torch.utils.data.distributed.DistributedSampler(val_datasets[key],shuffle=False,)
            val_loaders[key] = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, num_workers=config["num_workers"], pin_memory=True,sampler=valid_sampler[key],
            )
        else:
            val_loaders[key] = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, num_workers=config["num_workers"], pin_memory=True,
            )

    if config["wandb"]["flag"]:
        if config['ddp']==False or (config['ddp'] and local_rank==0):
            run = wandb.init(
                project=config["wandb"]["project_name"],
                name=config["name"]+f'_{split}' if num_splits > 1 else config["name"],
                reinit=True,
                config=config,
            )
    if config['ddp']==False or (config['ddp'] and local_rank==0):  
        print(model)  


    bests = {}
    for key in val_loaders:
        bests[key] = -1,-1,-1,1000
        path = config["test_load_path"]
        print(path)
        state_dict = torch.load(path, map_location=device, weights_only=False)["state_dict"]
        i_state_dict = OrderedDict()
        for state_dict_key in state_dict.keys():
            if "module" in state_dict_key:
                i_state_dict[state_dict_key[7:]] = state_dict[state_dict_key]
            else:
                i_state_dict[state_dict_key] = state_dict[state_dict_key]
        model.load_state_dict(i_state_dict, strict=False)

        if config['ddp'] == True:
        # DistributedDataParallel
            model.to(device)
            model= torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model.to(device)
        #model.eval()

        profile_inference(val_dataset, model, device)

        bests[key] = inference_set(
            val_loaders[key],
            model,
            device, bests[key][0:4], wandb_flag=config["wandb"]["flag"], ddp_flag=config['ddp'], save_model=config["save_model"], save_name=config["name"],
            suffix=key+"_s",
        )
        if config['ddp']==False or (config['ddp'] and local_rank==0):
            print(
                f"""For the finetuning process on {key} with {len(val_loaders[key])} videos,
                the best validation accuracy of the model-s is as follows:
                SROCC: {bests[key][0]:.4f}
                PLCC:  {bests[key][1]:.4f}
                KROCC: {bests[key][2]:.4f}
                RMSE:  {bests[key][3]:.4f}."""
            )
    
    profile_inference(val_dataset, model, device)
    if config["wandb"]["flag"]:
        if config['ddp']==False or (config['ddp'] and local_rank==0):
            run.finish()



if __name__ == "__main__":
    main()