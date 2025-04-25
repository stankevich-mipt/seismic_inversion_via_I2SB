# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import argparse

import torch
from pathlib import Path

from tqdm.auto import tqdm as tqdm
from torch.multiprocessing import Process
from torch.utils.data import DataLoader

from logger import Logger
import distributed_util as dist_util
from corruption import build_corruption
from dataset import openfwi
import utils.script as script_utils


def get_sample_dir(opt):

    sample_dir_prefix = opt.result_dir / opt.name / "samples" / opt.ckpt / opt.dataset_name / str(opt.seed) 
    sample_dir_postfix = "ode" if opt.ot_ode else "sde"
    sample_dir_postfix += f"_nfe={opt.nfe}"+\
                          f"_corrupt={opt.corrupt}" +\
                          (f"_guidance-scale={str(opt.guidance_scale)}" if opt.guidance_scale is not None else "") +\
                          (f"_clip" if opt.clip_denoise else "")
    
    sample_dir = sample_dir_prefix / sample_dir_postfix
    os.makedirs(sample_dir, exist_ok=True)

    return sample_dir


@torch.no_grad()
def main(opt):
    
    log = Logger(opt.global_rank, ".log")

    corrupt_type = opt.corrupt
    # build corruption method
    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)

    # build validation dataset
    val_dataset = openfwi.build_lmdb_dataset(opt, log, train=False)
    if opt.partition is not None:
        val_dataset = script_utils.build_partition(opt, val_dataset, log)        
    
    # set shuffle=True to make the output seed-dependant
    val_loader = DataLoader(val_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    sample_dir = get_sample_dir(opt)
    log.info(f"Recon images will be saved to {sample_dir}!")
    opt.sample_dir = sample_dir
    
    runner = script_utils.get_runner(opt, log, save_opt=False)
    img_collected = runner.sample(opt, val_loader, corrupt_method)

    log.info(f"Sampling complete! Collected {img_collected} recon images!")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--master-port",    type=int,  default=6020)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default="localhost", help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--name",           type=str,  required=True,       help="experiment ID")
    parser.add_argument("--result-dir",     type=Path, required=True,       help="root directory for all of the output files produced with the training script")
    parser.add_argument("--dataset-dir",    type=Path, default=None,        help="Use the dataset provided by the argument instead of the checkpointed one")
    parser.add_argument("--dataset-name",   type=str,  default=None,        help="Use the dataset metadata provided by the argument instead of the checkpointed one")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--ckpt",           type=str,  required=True,       help="the checkpoint name from which we wish to sample")
    parser.add_argument("--corrupt",        type=str,  default=None,        help="restoration task")
    parser.add_argument("--batch-size",     type=int,  default=32)
    parser.add_argument("--total-batches",  type=int,  default=100,         help="upper limit for total batches sampled during inference")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")
    parser.add_argument("--deterministic",  action="store_true",            help="use deterministic sampling during inference")
    parser.add_argument(
        "--stochastic",
        action="store_false", 
        dest="deteministic",
        help="use stochastic sampling during inference"
    )
    parser.add_argument(
        "--test-var-reduction", 
        action="store_true",
        help=(
            "register inversion results for batches of smooth models obtained"
            "through application of degradation operator to the same reference model"
        )
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=None,
        help="average unconditional and conditional network predictions during inference"
    )
    parser.set_defaults(stochastic=True)

    opt = script_utils.setup_inference_options(parser)
    script_utils.set_seed(opt.seed)    
    
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
