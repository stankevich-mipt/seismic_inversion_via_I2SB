# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from ddrm.
#
# Source:
# https://github.com/bahjat-kawar/ddrm/blob/master/functions/svd_replacement.py#L397
# https://github.com/bahjat-kawar/ddrm/blob/master/runners/diffusion.py#L245
# https://github.com/bahjat-kawar/ddrm/blob/master/runners/diffusion.py#L251
#
# The license for the original version of this file can be
# found in this directory (LICENSE_DDRM).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import numpy as np
import torch
from .base import H_functions
from scipy.ndimage import gaussian_filter


class GaussMixtureBlur():
    
    def __init__(
        self, 
        gauss_kernel_floor=8, 
        gauss_kernel_ceil=24, 
        noise_mixin_floor=0., 
        noise_mixin_ceil=0.3
    ):
        
        self.gauss_kernel_floor = gauss_kernel_floor
        self.gauss_kernel_ceil  = gauss_kernel_ceil
        self.noise_mixin_floor  = noise_mixin_floor
        self.noise_mixin_ceil   = noise_mixin_ceil
    
    def __call__(self, img):

        assert len(img.shape) == 4

        np_img = img.clone().detach().cpu().numpy()
        
        gauss_kernel_size = np.random.randint(self.gauss_kernel_floor, self.gauss_kernel_ceil+1)
        gauss_kernel = list([0, 0 , gauss_kernel_size, gauss_kernel_size])

        noise_mixin_level = self.noise_mixin_floor + self.noise_mixin_ceil * np.random.rand()

        np_img = np.random.normal(0., 1., np_img.shape) * noise_mixin_level + np_img * (1. - noise_mixin_level)
        np_img = gaussian_filter(np_img, gauss_kernel)
        np_img = np.clip(np_img, -1., 1.)

        img = torch.tensor(np_img).to(dtype=img.dtype, device=img.device)

        return img


class Deblurring(H_functions):

    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, ZERO = 3e-2):
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small[i, j] = kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        #ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1), self.singulars_small.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


def _blur_factory(opt, kernel):

    xdim = (1, opt.image_size, opt.image_size)

    if kernel == "uni":    
        uni = Deblurring(torch.Tensor([1/9] * 9).to(opt.device), 1, opt.image_size, opt.device)
        blur = lambda img: uni.H(img).reshape(img.shape[0], *xdim)
    elif kernel == "gauss":
        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        g_kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(opt.device)
        gauss = Deblurring(g_kernel / g_kernel.sum(), 3, opt.image_size, opt.device)
        blur = lambda img: gauss.H(img).reshape(img.shape[0], *xdim)
    elif kernel == "openfwi_custom":
        openfwi_blur = GaussMixtureBlur()
        blur = lambda img: openfwi_blur(img)
    elif kernel == "openfwi_benchmark":
        # blur parameters from https://arxiv.org/pdf/2410.21776
        openfwi_blur = GaussMixtureBlur(gauss_kernel_floor=9, gauss_kernel_ceil=9, noise_mixin_floor=0., noise_mixin_ceil=0.)
        blur = lambda img: openfwi_blur(img)
    else:
        raise NotImplementedError
           
    return blur


def build_blur(opt, log, kernel):

    log.info(f"[Corrupt] Bluring {kernel=}...")

    blurring_func = _blur_factory(opt, kernel)
    
    def blur(img):   
        img = (img + 1) / 2 # [-1,1] -> [0,1]
        img = blurring_func(img)
        img = img * 2 - 1 # [0,1] -> [-1,1]
        return img

    return blur