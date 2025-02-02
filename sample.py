# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import pickle
import argparse
import random

import torch
import numpy as np
import torch.distributed as dist
from pathlib import Path
from easydict import EasyDict as edict

from tqdm.auto import tqdm as tqdm
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner
from corruption import build_corruption
from dataset import imagenet, openfwi


def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu=}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset


def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)


def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx=}, {end_idx=}! Now size={len(subset)}!")
    return subset


def build_val_dataset(opt, log, corrupt_type):
    
    if "sr4x" in corrupt_type:
        val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False) # full 50k val
    elif "inpaint" in corrupt_type:
        mask = corrupt_type.split("-")[1]
        val_dataset = imagenet.InpaintingVal10kSubset(opt, log, mask) # subset 10k val + mask
    elif "blur" in corrupt_type:
        kernel = corrupt_type.split("-")[1]
        if kernel == "openfwi_custom":
            val_dataset = openfwi.build_lmdb_dataset(opt, log, train=False)
        else:
            val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
    elif corrupt_type == "mixture":
        from corruption.mixture import MixtureCorruptDatasetVal
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
        val_dataset = MixtureCorruptDatasetVal(opt, val_dataset) # subset 10k val + mixture
    else:
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log) # subset 10k val

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset


def get_sample_dir(opt):

    sample_dir_prefix = opt.result_dir / opt.name / "samples" / str(opt.seed) 
    sample_dir_postfix = f"nfe={opt.nfe}"+\
                         f"_guidance-scale={str(opt.guidance_scale) if opt.guidance_scale else "0.0"}" +\
                         f"_{"clip" if opt.clip_denoise else ""}"
    
    sample_dir = sample_dir_prefix / sample_dir_postfix
    os.makedirs(sample_dir, exist_ok=True)

    return sample_dir


def compute_batch(opt, corrupt_type, corrupt_method, out):

    if "inpaint" in corrupt_type:
        clean_img, y, mask = out
        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
    elif corrupt_type == "mixture":
        clean_img, corrupt_img, y = out
        mask = None
    else:
        clean_img, y = out
        mask = None
        corrupt_img = corrupt_method(clean_img.to(opt.device))
        
        clean_img = clean_img[[0],].repeat(y.shape[0], 1, 1, 1)
        y = y[[0],].repeat(clean_img.shape[0], 1, 1, 1)
        mask = None
        corrupt_img = torch.empty_like(clean_img)
        
        for j in range(corrupt_img.shape[0]):
            corrupt_img[j] = corrupt_method(clean_img[[j,]].to(opt.device)).squeeze(0)
        
        x1 = corrupt_img.to(opt.device)

    x0 = clean_img.detach().to(opt.device)

    cond = y.detach() if opt.cond_y else None
    return x0, x1, cond


@torch.no_grad()
def main(opt):
    
    log = Logger(opt.global_rank, ".log")

    corrupt_type = opt.corrupt
    nfe  = opt.nfe 
    # build corruption method
    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)

    # build imagenet val dataset
    val_dataset = build_val_dataset(opt, log, corrupt_type)
    
    # set shuffle=True to make the output seed-dependant
    val_loader = DataLoader(val_dataset,
        batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    runner = Runner(opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    opt.cond_y = True 
    opt.ot_ode = True

    # create save folder
    sample_dir = get_sample_dir(opt)
    log.info(f"Recon images will be saved to {sample_dir}!")

    img_collected = 0

    for loader_itr, out in enumerate(val_loader):

        if loader_itr == opt.total_batches: break

        _, x1, cond = compute_batch(opt, corrupt_type, corrupt_method, out)
        xs, _ = runner.ddpm_sampling(opt, x1, cond, nfe=nfe, verbose=False)
        
        recon_img = xs[:, 0, ...].to(opt.device)
        cond = cond.to(opt.device)
        corrupt_img = x1.detach()
        assert recon_img.shape == corrupt_img.shape

        if loader_itr == 0 and opt.global_rank == 0: # debug
            
            os.makedirs(".debug", exist_ok=True)
            tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png")
            tu.save_image((recon_img+1)/2, ".debug/recon.png")
            log.info("Saved debug images!")
        
        gathered_recon_img = collect_all_subset(recon_img, log=None)
        gathered_cond = collect_all_subset(cond, log=None)

        if opt.global_rank == 0:

            save_path = sample_dir/ f"batch_{loader_itr}.pt"
            torch.save({"model": gathered_recon_img, "data": gathered_cond}, save_path)
       
        img_collected += len(gathered_recon_img)
        dist.barrier()

    log.info(f"Sampling complete! Collected {img_collected} recon images!")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--master-port",    type=int,  default=6020)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--name",           type=str,  required=True,       help="experiment ID")
    parser.add_argument("--result-dir",     type=Path, required=True,       help="root directory for all of the output files produced with the training script")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=32)
    parser.add_argument("--total-batches",  type=int,  default=100,         help="upper limit for total batches sampled during inference")
    parser.add_argument("--ckpt",           type=str,  default="latest.pt", help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=None,
        help="average unconditional and conditional network predictions during inference"
    )


    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device=torch.device("cuda:0"),
        clip_denoise=False,
        use_fp16=False,
    )
    opt.update(vars(arg))

    
    # restore missing keys from option checkpoint 
    opt_pkl_path = opt.result_dir / opt.name / "checkpoints" / "options.pkl" 
    
    with open(opt_pkl_path, "rb") as f:
        ckpt_opt = pickle.load(f)
    arg_dict = vars(arg)
    
    for k, v in copy.copy(vars(ckpt_opt)).items():
        if arg_dict.get(k, 0):
            del ckpt_opt.__dict__[k]  

    opt.update(vars(ckpt_opt))

    # setup checkpoint path
    opt.load = opt.result_dir / opt.name / "checkpoints" / opt.ckpt
    
    # set seed
    set_seed(opt.seed)    
    
    if opt.distributed:
    
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
