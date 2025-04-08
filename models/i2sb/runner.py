# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import numpy as np
from .network import UNet
from .diffusion import Diffusion
from models.runner import make_beta_schedule, BaseRunner


class Runner(BaseRunner):

    def __init__(self, opt, log, save_opt=True, betas=None):

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])

        super().__init__(opt, log, save_opt, model=UNet, diffusion=Diffusion, betas=betas)

    def prepare_training_signature(self, opt, dataloader, corrupt_method):
    
        # sample boundary pair along with condition 
        c0, c1, cond = self.get_data_triplet(opt, dataloader, corrupt_method)
        # uniformly sample training timesteps
        step  = torch.randint(0, opt.interval, (c0.shape[0],))
        # corrupt the sample from target distribution with noise
        ct    = self.diffusion.q_sample(step, c0, c1, ot_ode=opt.ot_ode)
        # compute regression label
        label = self.compute_label(step, c0, ct, pred_c0=opt.pred_c0)

        if self.cond:
            # mask some of conditional inputs with zeros - nesessary for classifier-free guidance
            keep_cond = (torch.rand(cond.shape[0], 1, 1, 1) < (1. - opt.drop_cond)).type(cond.dtype).to(cond.device)
            cond = cond * keep_cond
            # assign higher weights for samples with non-zero condition
            weights = 1. + keep_cond * 10.
        else:
            cond = None  
            weights = torch.ones(c0.shape[0], 1, 1, 1).type(c0.dtype).to(c0.device)

        weights = weights.repeat(1, *label.shape[1:])

        return step, ct, cond, label, weights

    def compute_label(self, step, c0, ct, pred_c0=False):

        if not pred_c0:
            std_fwd = self.diffusion.get_std_fwd(step, xdim=c0.shape[1:])
            label = (ct - c0) / std_fwd
        else:
            label = c0
        return label.detach()
    
    def compute_pred_c0(self, step, ct, net_out, pred_c0=False, clip_denoise=False):

        if not pred_c0:
            std_fwd = self.diffusion.get_std_fwd(step, xdim=ct.shape[1:])
            pred_c0 = ct - std_fwd * net_out
        else:
            pred_c0 = net_out
        if clip_denoise: pred_c0.clamp_(-1., 1.)
        
        return pred_c0