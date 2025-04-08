# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import math
import numpy as np
import torch

from .network import UNet
from .diffusion import Diffusion
from models.util import unsqueeze_xdim
from models.runner import BaseRunner


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class Runner(BaseRunner):

    def __init__(self, opt, log, save_opt=True):

        betas = get_named_beta_schedule('cosine', opt.interval)

        super().__init__(opt, log, save_opt, model=UNet, diffusion=Diffusion, betas=betas)

    def compute_label(self, step, c0, ct, pred_c0=False):

        if not pred_c0:
            mu_c0 = unsqueeze_xdim(self.diffusion.sqrt_alphas_cumprod[step], ct.shape[1:])
            std_noise = unsqueeze_xdim(self.diffusion.sqrt_one_minus_alphas_cumprod[step], ct.shape[1:])
            label = (ct - mu_c0 * c0) / std_noise
        else:
            label = c0

        return label.detach()
    
    def compute_pred_c0(self, step, ct, net_out, pred_c0=False, clip_denoise=False):

        if not pred_c0:
            mu_c0 = unsqueeze_xdim(self.diffusion.sqrt_alphas_cumprod[step], ct.shape[1:])
            std_noise = unsqueeze_xdim(self.diffusion.sqrt_one_minus_alphas_cumprod[step], ct.shape[1:])
            pred_c0 = (ct - std_noise * net_out) / mu_c0
        else:
            pred_c0 = net_out
        if clip_denoise: pred_c0.clamp_(-1., 1.)
        
        return pred_c0 

    def prepare_training_signature(self, opt, dataloader, corrupt_method):
        """
        Method that aggregates all the nesessary manipulations  
        with model inputs and outputs during training 
        """
        
        # sample boundary pair along with condition 
        c0, c1, seismic_data = self.get_data_triplet(opt, dataloader, corrupt_method)
        # uniformly sample training timesteps
        step  = torch.randint(0, opt.interval, (c0.shape[0],))
        # corrupt the sample from target distribution with noise
        ct    = self.diffusion.q_sample(step, c0)
        # compute regression label
        label = self.compute_label(step, c0, ct, pred_c0=opt.pred_c0)

        # conditional input is always present for current implementation
        # if self.cond=True, concatenate seismic data with smooth velocity model
        cond = c1
        if self.cond:
            # mask some of conditional inputs with zeros - nesessary for classifier-free guidance
            cond = torch.cat([cond, seismic_data], dim=1).detach()
        
        keep_cond = (torch.rand(cond.shape[0], 1, 1, 1) < (1. - opt.drop_cond)).type(cond.dtype).to(cond.device)
        cond = cond * keep_cond

        weights = torch.ones(c0.shape[0], 1, 1, 1).type(c0.dtype).to(c0.device).repeat(1, *label.shape[1:])

        return step, ct, cond, label, weights

