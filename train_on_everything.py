# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import argparse

import copy
from pathlib import Path

import torch
from torch.multiprocessing import Process

from logger import Logger
from distributed_util import init_processes
from corruption import build_corruption
from dataset import openfwi, MultiLMDBDataset
import utils.script as script_utils


def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--master-port",    type=int,   default=6020)
    parser.add_argument("--name",           type=str,   default=None,        help="experiment ID")
    parser.add_argument("--ckpt",           type=str,   default=None,        help="resumed checkpoint name")
    parser.add_argument("--gpu",            type=int,   default=None,        help="set only if you wish to run on a particular device")
    parser.add_argument("--n-gpu-per-node", type=int,   default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,   default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,   default=1,           help="The number of nodes in multi node env")

    # --------------- SB model ---------------
    parser.add_argument("--model",          type=str,   default="unet_ch32", help="model architecture")
    parser.add_argument("--image-size",     type=int,   default=256)
    parser.add_argument("--corrupt",        type=str,   default=None,        help="restoration task")
    parser.add_argument("--val-batches",    type=int,   default=100)
    parser.add_argument("--t0",             type=float, default=1e-4,        help="sigma start time in network parametrization")
    parser.add_argument("--T",              type=float, default=1.,          help="sigma end time in network parametrization")
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--beta-max",       type=float, default=0.3,         help="max diffusion for the diffusion model")
    parser.add_argument("--drop_cond",      type=float, default=0.25,        help="probability to replace conditional input with zero-valued tensor")
    parser.add_argument("--pred_c0",        action="store_true",             help="predict bridge starting point instead of scaled noise")       
    parser.add_argument("--ot-ode",         action="store_true",             help="use OT-ODE model")
    parser.add_argument("--clip-denoise",   action="store_true",             help="clamp predicted image to [-1,1] at each")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--microbatch",     type=int,   default=2,           help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--num-itr",        type=int,   default=1000000,     help="training iteration")
    parser.add_argument("--lr",             type=float, default=5e-5,        help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.99,        help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,        help="learning rate decay step size")
    parser.add_argument("--weight-decay",   type=float, default=0.0)
    parser.add_argument("--clip-grad-norm", type=float, default=None,        help="Clip gradent norm value up to a set value")
    parser.add_argument("--ema",            type=float, default=0.99)

    # --------------- path and logging ---------------
    parser.add_argument("--result-dir",     type=Path,  required=True,       help="root directory for all of the output files produced with the training script")
    parser.add_argument("--dataset-dir",    type=Path,  default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--loss-log-freq",  type=int,   default=10,          help="register loss value every 'loss-log-freq' batches")
    parser.add_argument("--save-freq",      type=int,   default=5000,        help="save model and optimizer checkpoints every 'save-freq' batches")
    parser.add_argument("--val-freq",       type=int,   default=5000,        help="run model validation every 'val-freq' batches")
    parser.add_argument("--log-writer",     type=str,   default=None,        help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",  type=str,   default=None,        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,        help="user name of your W&B account")
    parser.add_argument(
        "--json-data-config", 
        type=Path, 
        default="/home/seismic_inversion_via_I2SB/dataset/config/openfwi_dataset_config.json", 
        help="full path to JSON config for openfwi datasets"
    )

    opt = parser.parse_args()

    # ========= auto setup =========
    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    if opt.name is None:
        opt.name = opt.corrupt
    opt.distributed = opt.n_gpu_per_node > 1

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========

    RESULT_DIR = opt.result_dir
    
    opt.ckpt_path = RESULT_DIR / str(opt.name) / "checkpoints"
    opt.log_dir   = RESULT_DIR / str(opt.name) / "logs"

    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_file = opt.ckpt_path / opt.ckpt
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    # ========= auto assert =========
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size=} is not dividable by {opt.microbatch}!"
    return opt

def main(opt):
    
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("              Seismic Inversion via I2SB               ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        script_utils.set_seed(opt.seed + opt.global_rank)

    # build imagenet dataset

    all_datasets = (
        'FlatVel_A', 
        'FlatVel_B', 
        'CurveVel_A', 
        'CurveVel_B', 
        'FlatFault_A', 
        'FlatFault_B', 
        'CurveFault_A',
        'CurveFault_B', 
        'Style_A', 
        'Style_B',      
    )

    root_dataset_dir = opt.dataset_dir

    lmdb_train_datasets = []
    lmdb_val_datasets = []

    for dataset in all_datasets:

        opt.dataset_name = dataset
        opt.dataset_dir  = root_dataset_dir / dataset

        train_dataset = openfwi.build_lmdb_dataset(opt, log, train=True)
        val_dataset   = openfwi.build_lmdb_dataset(opt, log, train=False)

        lmdb_train_datasets.append(train_dataset)
        lmdb_val_datasets.append(val_dataset)


    train_dataset = MultiLMDBDataset(lmdb_train_datasets)
    val_dataset   = MultiLMDBDataset(lmdb_val_datasets)

    # build corruption method
    corrupt_method = build_corruption(opt, log)

    model_name = opt.model
    if "ddpm" in model_name:
        from models.ddpm import Runner
    elif "inversionnet" in model_name:
        from models.inversionnet import Runner
    elif "i2sb" in model_name:
        from models.i2sb import Runner
    else:
        raise NotImplementedError
        
    run = Runner(opt, log, save_opt=True)  
    # Set sampling mode alias for disambiguation 
    # Deterministic is related to inference, ot_ode is related to training
    # For model trained with stochastic algorithm deterministic inference still makes sense
    # Reverse is not true, hence ot_ode is on the left side of assignment
    opt.deterministic = opt.ot_ode
    
    run.train(opt, train_dataset, val_dataset, corrupt_method)
    log.info("Finish!")

if __name__ == '__main__':

    opt = create_training_options()

    assert opt.corrupt is not None

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
            p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        init_processes(0, opt.n_gpu_per_node, main, opt)