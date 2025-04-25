import os
import random
import numpy as np
import torch
import pickle
import copy

from easydict import EasyDict as edict
from torch.utils.data import Subset
from torch_ema import ExponentialMovingAverage


def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


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


def setup_inference_options(argparser):

    arg = argparser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device=torch.device("cuda:0"),
        clip_denoise=False,
        use_fp16=False,
        eval_on_train=False
    )
    opt.update(vars(arg))

    opt_pkl_path = opt.result_dir / opt.name / "checkpoints" / "options.pkl" 
    
    with open(opt_pkl_path, "rb") as f:
        ckpt_opt = pickle.load(f)

    arg_dict = vars(arg)
    
    for k, _ in copy.copy(vars(ckpt_opt)).items():
        if arg_dict.get(k, 0):
            del ckpt_opt.__dict__[k]

    opt.update(vars(ckpt_opt))

    # in case if cI2SB model was trained as ot-ode flow matching one, 
    # stochastic sampling  is not applicable 
    if opt.ot_ode: opt.deterministic = True 
    
    # setup checkpoint path
    opt.load = opt.result_dir / opt.name / "checkpoints" / opt.ckpt

    return opt


def get_runner(opt, log, save_opt=False):

    model_name = opt.model
    if "ddpm" in model_name:
        from models.ddpm import Runner
    elif "inversionnet" in model_name:
        from models.inversionnet import Runner
    elif "i2sb" in model_name:
        from models.i2sb import Runner
    else:
        raise NotImplementedError  

    runner = Runner(opt, log, save_opt=save_opt)  
    
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        if "inversionnet" in model_name:
            runner.net.model.convert_to_fp16()
        else: 
            runner.net.diffusion_model.convert_to_fp16()
        # re-init ema with fp16 weight
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) 
    
    return runner
