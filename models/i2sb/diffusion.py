# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from ipdb import set_trace as debug

from models.util import unsqueeze_xdim


def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


class Diffusion():
    
    def __init__(self, betas, device):

        self.device = device

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_c0, mu_c1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_c0 = to_torch(mu_c0).to(device)
        self.mu_c1 = to_torch(mu_c1).to(device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, c0, c1, ot_ode=False):

        assert c0.shape == c1.shape
        batch, *xdim = c0.shape

        mu_c0  = unsqueeze_xdim(self.mu_c0[step],  xdim)
        mu_c1  = unsqueeze_xdim(self.mu_c1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        ct = mu_c0 * c0 + mu_c1 * c1
        if not ot_ode:
            ct = ct + std_sb * torch.randn_like(ct)
        return ct.detach()

    def p_posterior(self, nprev, n, c_n, c0, ot_ode=False):

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_c0, mu_cn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        ct_prev = mu_c0 * c0 + mu_cn * c_n
        if not ot_ode and nprev > 0:
            ct_prev = ct_prev + var.sqrt() * torch.randn_like(ct_prev)

        return ct_prev

    def ddpm_sampling(self, steps, pred_c0_fn, smooth_model, ot_ode=False, log_steps=None, verbose=True):
        
        ct = smooth_model.clone().detach().to(self.device)

        traj_ct = []
        traj_c0 = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        
        for prev_step, step in pair_steps:
            
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_c0  = pred_c0_fn(ct, step)
            ct = self.p_posterior(prev_step, step, ct, pred_c0, ot_ode=ot_ode)

            if prev_step in log_steps:
                traj_c0.append(pred_c0.detach())
                traj_ct.append(ct.detach())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(traj_ct), stack_bwd_traj(traj_c0)
