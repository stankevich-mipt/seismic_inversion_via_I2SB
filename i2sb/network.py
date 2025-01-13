# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import torch

from guided_diffusion.script_util import create_model

from . import util
from .ckpt_util import (
    I2SB_IMG256_UNCOND_PKL,
    I2SB_IMG256_UNCOND_CKPT,
    I2SB_IMG256_COND_PKL,
    I2SB_IMG256_COND_CKPT,
)

class Image256Net(torch.nn.Module):
    
    def __init__(self, log, opt, noise_levels):
        
        super(Image256Net, self).__init__()

        # checkpoint with options is created prior to the net instance        
        ckpt_pkl = os.path.join(opt.ckpt_path, 'options.pkl')
        with open(ckpt_pkl, "rb") as f:
            kwargs = pickle.load(f)
    
        self.diffusion_model = create_model(**kwargs)
        log.info(f"[Net] Initialized network from {ckpt_pkl=}! Size={util.count_parameters(self.diffusion_model)}!")

        self.diffusion_model.eval()
        self.cond = opt.cond_y
        self.noise_levels = noise_levels

    def forward(self, x, steps, cond):

        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        x = torch.cat([x, cond], dim=1) if self.cond else x
        return self.diffusion_model(x, t)
