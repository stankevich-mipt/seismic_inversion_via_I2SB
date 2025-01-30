# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch

from .import util
from guided_diffusion.unet import UNetModel
from config import get_config_by_name


class Image256Net(torch.nn.Module):

    def __init__(self, log, opt, noise_levels):
        
        super(Image256Net, self).__init__()

        model_config = get_config_by_name(opt.model)
        model_config.image_size = opt.image_size
    
        self.diffusion_model = UNetModel(**model_config)

        log.info(f"[Net] Created network with {opt.model} architecture! Size={util.count_parameters(self.diffusion_model)}!")

        self.diffusion_model.eval()
        
        self.noise_levels = noise_levels

    def forward(self, x, steps, cond):

        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        x = torch.cat([x, cond], dim=1)
        return self.diffusion_model(x, t)
