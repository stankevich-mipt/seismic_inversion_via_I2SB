# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from .network import UNet
from .diffusion import Diffusion
from models.runner import make_beta_schedule, BaseRunner


class Runner(BaseRunner):

    def __init__(self, opt, log, save_opt=True, betas=None):

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])

        super().__init__(opt, log, save_opt, model=UNet, diffusion=Diffusion, betas=betas)

    def compute_label(self, step, x0, xt, pred_x0=False):

        if not pred_x0:
            std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
            label = (xt - x0) / std_fwd
        else:
            label = x0
        return label.detach()
    
    def compute_pred_x0(self, step, xt, net_out, pred_x0=False, clip_denoise=False):

        if not pred_x0:
            std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
            pred_x0 = xt - std_fwd * net_out
        else:
            pred_x0 = net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        
        return pred_x0

