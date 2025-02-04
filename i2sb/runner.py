# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as tu
import distributed_util as dist_util

from tqdm.auto import tqdm as tqdm
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage

from . import util
from utils.pytorch_ssim import SSIM
from .network import Image256Net
from .diffusion import Diffusion


def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.weight_decay}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()


def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()


class Runner(object):
    
    def __init__(self, opt, log, save_opt=True):
        
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image256Net(log, opt, noise_levels)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

    def compute_label(self, step, x0, xt, pred_x0=False):
        """ Eq 12 """
        if not pred_x0:
            std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
            label = (xt - x0) / std_fwd
        else:
            label = x0
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, pred_x0=False, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        if not pred_x0:
            std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
            pred_x0 = xt - std_fwd * net_out
        else:
            pred_x0 = net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def process_batch(self, opt, batch, corrupt_method, augment=False):
        
        clean_img, cond = batch
        with torch.no_grad():
            corrupt_img = corrupt_method(clean_img.to(opt.device))

        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)

        cond = cond.detach().to(opt.device)
        
        if augment:
            # occasionally replace conditional input with zeros
            cond = cond * (torch.rand(cond.shape[0], 1, 1, 1) < (1. - opt.drop_cond)).type(cond.dtype).to(cond.device)
    
        assert x0.shape == x1.shape
        return x0, x1, cond

    def sample_batch(self, opt, loader, corrupt_method, augment=False): 
        return self.process_batch(opt, next(loader), corrupt_method, augment=augment)

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, cond = self.sample_batch(opt, train_loader, corrupt_method, augment=True)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt, pred_x0=opt.pred_x0)


                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape

                loss = F.mse_loss(pred, label)
                loss.backward()

            if opt.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), opt.clip_grad_norm)

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 1000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it % 500 == 0:
                net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(
        self, opt, x1, cond, 
        nfe=None, log_count=None, verbose=True
    ):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        if log_count is None: log_count = len(steps)-1

        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0

        if verbose: self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1   = x1.to(opt.device)
        cond = cond.to(opt.device)
        
        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):

                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)

                # If guidance scale is present, apply formula 6 from https://arxiv.org/abs/2207.12598
                guidance_scale = getattr(opt, 'guidance_scale', 0)
                
                if not guidance_scale:
                    out = self.net(xt, step, cond=cond)
                else:
                    out_cond   = self.net(xt, step, cond)
                    out_uncond = self.net(xt, step, 0.*cond)
                    out = (1 + guidance_scale) * out_cond - guidance_scale * out_uncond
                 
                return self.compute_pred_x0(step, xt, out, pred_x0=opt.pred_x0, clip_denoise=opt.clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, log_steps=log_steps, verbose=verbose, opt_ode=opt.opt_ode
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        img_clean, img_corrupt, cond = self.sample_batch(opt, val_loader, corrupt_method, augment=False)

        x1 = img_corrupt.to(opt.device)

        xs, pred_x0s = self.ddpm_sampling(opt, x1, cond, log_count=10)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        # pick shot #3 for display
        img_cond = cond[:, [2]]

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        img_cond    = all_cat_cpu(opt, log, img_cond)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        log.info("Logging images ...")
        
        img_recon = xs[:, 0, ...]
        
        log_image("image/clean",   img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/cond",    img_cond)
        log_image("image/recon",   img_recon)
        
        log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)

        log.info(f"Evaluating metrices on {opt.val_batches} validation batches at NFE=1 ...")   
        
        val_opt = copy.deepcopy(opt)
        val_opt.ot_ode = True
        val_opt.guidance_scale = None

        avg_mse  = 0.
        avg_mae  = 0.
        avg_ssim = 0.
        total_samples = 0.

        l1   = torch.nn.L1Loss(reduction='mean')
        l2   = torch.nn.MSELoss(reduction='mean')
        ssim = SSIM(window_size=11)

        pbar = tqdm(val_loader) 
        processed_batches = int(0)

        for batch in pbar:

            if processed_batches == opt.val_batches: break

            x0, x1, cond = self.process_batch(val_opt, batch, corrupt_method)

            xs, pred_x0s = self.ddpm_sampling(val_opt, x1, cond=cond, nfe=1, verbose=False)
            
            img_recon = xs[:, 0, ...].to(val_opt.device)
            img_recon = all_cat_cpu(opt, log, img_recon)
            img_clean = all_cat_cpu(opt, log, x0)

            avg_mae  += l1(img_recon, img_clean) * xs.shape[0]
            avg_mse  += l2(img_recon, img_clean) * xs.shape[0]
            avg_ssim += ssim(img_recon/2. + 0.5, img_clean/2. + 0.5) * xs.shape[0]

            total_samples += xs.shape[0]
            processed_batches += 1

        avg_mse  /= total_samples
        avg_mae  /= total_samples
        avg_ssim /= total_samples

        self.writer.add_scalar(it, "MSE", avg_mse)
        self.writer.add_scalar(it, "MAE", avg_mae)
        self.writer.add_scalar(it, "SSIM", avg_ssim)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
