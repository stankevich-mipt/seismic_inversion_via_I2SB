# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch

from .network import UNet
from .diffusion import Diffusion
from models.util import unsqueeze_xdim
from models.runner import BaseRunner


class Runner(BaseRunner):

    def __init__(self, opt, log, save_opt=True, betas=None):

        super().__init__(opt, log, save_opt, model=UNet, diffusion=Diffusion, betas=betas)

    def compute_label(self, step, x0, xt, pred_x0=False):
        """ Eq 12 """
        if not pred_x0:
            mu_x0 = unsqueeze_xdim(self.diffusion.sqrt_alphas_cumprod[step], xt.shape[1:])
            std_noise = unsqueeze_xdim(self.diffusion.sqrt_one_minus_alphas_cumprod[step], xt.shape[1:])
            label = (xt - mu_x0 * x0) / std_noise
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
    
    def get_diffusion_endpoints(self, x0, corrupt_method):
        
        init_guess = corrupt_method(x0).clone().detach()
        noise = torch.randn_like(init_guess)
        return init_guess, noise

