# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import copy
import torch
import numpy as np
from tqdm import tqdm

from models.util import unsqueeze_xdim


class Diffusion():
    
    def __init__(self, betas, device):

        self.device = device

        # tensorize everything
      
        self.betas = torch.tensor(betas, dtype=torch.float32).to(device)
        self._get_posteriors_from_betas()
        
    def _get_posteriors_from_betas(self):

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]), 0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_mean_c0 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_c1 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _get_respaced_betas_from_alphas(self, timesteps_to_use):

        last_alpha_cumprod = 1.0
        new_betas = []

        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in timesteps_to_use:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
 
        self.betas = torch.tensor(new_betas, device=self.device)
        self._get_posteriors_from_betas()
                
    def q_sample(self, step, c0):
       
        _, *xdim = c0.shape
        
        mu_c0 = unsqueeze_xdim(self.sqrt_alphas_cumprod[step], xdim)
        std_noise = unsqueeze_xdim(self.sqrt_one_minus_alphas_cumprod[step], xdim)
        
        xt = mu_c0 * c0 + std_noise * torch.randn_like(c0) 

        return xt
    
    def p_posterior(self, step, c_n, c0, ot_ode=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        mu_c0 = self.posterior_mean_c0[step]
        mu_cn = self.posterior_mean_c1[step]
        var   = self.posterior_variance[step]

        ct_prev = mu_c0 * c0 + mu_cn * c_n
        if not ot_ode and step > 0:
            ct_prev = ct_prev + var.sqrt() * torch.randn_like(ct_prev)

        return ct_prev

    def ddpm_sampling(self, steps, pred_c0_fn, c1, ot_ode=False, log_steps=None, verbose=True):
        
        ct = torch.randn_like(c1).detach().to(self.device)

        ct_traj = []
        c0_traj = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        rescaled_diffusion = copy.deepcopy(self)
        rescaled_diffusion._get_respaced_betas_from_alphas(steps)

        step_mapping = dict(enumerate(steps))
        steps = list(step_mapping.keys())[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        
        for prev_step, step in pair_steps:

            pred_c0 = pred_c0_fn(torch.cat([ct, c1], dim=1), step_mapping[step])
            ct = rescaled_diffusion.p_posterior(prev_step, ct, pred_c0, ot_ode=ot_ode)
            
            if step_mapping[prev_step] in log_steps:
                c0_traj.append(pred_c0.detach())
                ct_traj.append(ct.detach())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))

        del rescaled_diffusion

        return stack_bwd_traj(ct_traj), stack_bwd_traj(c0_traj)
