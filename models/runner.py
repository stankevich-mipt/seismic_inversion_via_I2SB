import os
import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tu
import distributed_util as dist_util

from tqdm.auto import tqdm as tqdm
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader
        
from . import util
from utils.pytorch_ssim import SSIM


class MockModel(nn.Module):
    """
    mockup model class for signature consistency
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class MockDiffusion():
    """
    mockup diffusion class for signature consistency
    """

    def __init__(self, *args, **kwargs):
        pass

    def q_sample(self, *args, **kwargs):
        raise NotImplementedError
    
    def ddim_sampling(self, *args, **kwargs):
        raise NotImplementedError


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


def drop_shots_and_traces(data, p_drop_shots=0.5, p_drop_traces=0.5):
    
    drop_shots = (torch.rand(data.shape[0], data.shape[1], 1, 1) < p_drop_shots).type(torch.bool)
    # pick the item guaranteed to stay 
    nonzero_id_shots = torch.randint(data.shape[1], size=(data.shape[0],))
    drop_shots[
        torch.arange(data.shape[0]), 
        nonzero_id_shots, 
        torch.zeros(data.shape[0]).int(),
        torch.zeros(data.shape[0]).int()
    ] = torch.full((data.shape[0],), False)

    drop_traces = (torch.rand(data.shape[0], data.shape[1], 1, data.shape[3]) < p_drop_traces).type(torch.bool)
    # pick the item guaranteed to stay
    ids_batch, ids_shot = torch.meshgrid(torch.arange(data.shape[0]), torch.arange(data.shape[1]), indexing="ij")
    nonzero_id_traces = torch.randint(data.shape[3], size=(data.shape[0], data.shape[1]))
    drop_traces[
        ids_batch.flatten(), 
        ids_shot.flatten(),
        torch.zeros(data.shape[0] * data.shape[1]).int(), 
        nonzero_id_traces.flatten()
    ] = torch.full((data.shape[0] * data.shape[1],), False)

    data_mask = (1. - drop_shots.type(data.dtype)) * (1. - drop_traces.type(data.dtype))

    return data * data_mask.to(data.device)


class BaseRunner(object):

    def __init__(
        self, opt, log, 
        save_opt=True, 
        model=MockModel,
        diffusion=MockDiffusion, 
        betas=None
    ):

        super().__init__()

        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        if betas is None:
            betas = make_beta_schedule(n_timestep=opt.interval)

        self.diffusion = diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = model(log, opt, noise_levels)
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
        self.cond = "cond" in opt.model

    
    def process_batch(self, opt, batch, corrupt_method, augment=False):
        
        clean_img, cond = batch
        with torch.no_grad():
            corrupt_img = corrupt_method(clean_img.to(opt.device))

        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        cond = cond.detach().to(opt.device)

        if augment:
            x0, x1, cond = self.augment_batch(x0, x1, cond)

        assert x0.shape == x1.shape
        return x0, x1, cond
    
    def augment_batch(self, x0, x1, cond):
        """
        Augmentation in the base class does nothing
        """
        return x0, x1, cond

    def compute_label(self, step, x0, xt, pred_x0=False):
        """
        Framework-dependent regression label computation
        """
        raise NotImplementedError

    def compute_pred_x0(self, step, xt, net_out, pred_x0=False, clip_denoise=False):
        """
        Framework-dependent handling of neural net prediction
        """
        return NotImplementedError

    def sample_training_batch(self, opt, loader, corrupt_method, augment=False):

        x0, x1, cond = self.process_batch(opt, next(loader), corrupt_method, augment=augment)
        # mask some of conditional inputs with zeros - nesessary for classifier-free guidance
        keep_cond = (torch.rand(cond.shape[0], 1, 1, 1) < (1. - opt.drop_cond)).type(cond.dtype).to(cond.device)
        cond = cond * keep_cond 

        return x0, x1, cond


    def prepare_training_signature(self, opt, dataloader, corrupt_method):
        """
        Method that aggregates all the nesessary manipulations  
        with model inputs and outputs during training 
        """
        
        # sample boundary pair along with condition 
        x0, x1, cond = self.sample_training_batch(opt, dataloader, corrupt_method, augment=False)
        # uniformly sample training timesteps
        step  = torch.randint(0, opt.interval, (x0.shape[0],))
        # corrupt the sample from target distribution with noise
        xt    = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
        # compute regression label
        label = self.compute_label(step, x0, xt, pred_x0=opt.pred_x0)

        cond = cond if self.cond else None
        xt = torch.cat((xt, x1), dim=1)
        
        return step, xt, cond, label    

    def get_diffusion_endpoints(self, x0, corrupt_method):
        x1 = corrupt_method(x0)
        return x1, x1.clone().detach()

    def train(
        self, opt, train_dataset, val_dataset, corrupt_method,
        loss_log_freq=10, save_freq=1000, val_freq=500
    ):
        
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

                step, xt, cond, label = self.prepare_training_signature(
                    opt, train_loader, corrupt_method 
                )

                pred = net(xt, step, cond=cond)

                assert label.shape == pred.shape

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
            
            if it % loss_log_freq == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % save_freq == 0:
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

            if it % val_freq == 0:
                net.eval()
                self.validate(opt, it, val_loader, corrupt_method)
                net.train()
        
        self.writer.close()


    @torch.no_grad()
    def ddpm_sampling(
        self, opt, x1, init_guess, cond, 
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

        # x1 is the element of endpoint SDE distribution
        # init_guess is the smooth velocity model 

        init_guess = init_guess.to(opt.device) 
        cond = cond.to(opt.device) if self.cond else None
        x1 = x1.to(opt.device)
        
        with self.ema.average_parameters():
            
            self.net.eval()

            def pred_x0_fn(xt, step):

                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                xt = torch.cat((xt, init_guess), dim=1)
               
                guidance_scale = getattr(opt, 'guidance_scale', 0)
                
                if guidance_scale and (not self.cond):    
                    raise RuntimeError(
                        f"Unconditional models do not support guidance"
                        f"got guidance scale={guidance_scale}" 
                        f"while self.cond set to {self.cond}"
                    )

                # If guidance scale is present, apply formula 6 from https://arxiv.org/abs/2207.12598
                if not guidance_scale:
                    out = self.net(xt, step, cond=cond)
                else:
                    out_cond   = self.net(xt, step, cond)
                    out_uncond = self.net(xt, step, 0.*cond)
                    out = guidance_scale * out_cond + (1 - guidance_scale) * out_uncond

                return self.compute_pred_x0(step, xt, out, pred_x0=opt.pred_x0, clip_denoise=opt.clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, log_steps=log_steps, verbose=verbose, ot_ode=opt.ot_ode
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0


    @torch.no_grad()
    def validate(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        ground_truth, cond = next(val_loader)
        init_guess, x1 = self.get_diffusion_endpoints(ground_truth, corrupt_method)
        init_guess, x1 = init_guess.to(opt.device), x1.to(opt.device)
        
        xs, pred_x0s = self.ddpm_sampling(opt, x1, init_guess, cond, log_count=10)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        # pick shot #3 for display
        img_cond = cond[:, [2]]

        log.info("Collecting tensors ...")
        
        xs = all_cat_cpu(opt, log, xs)
        pred_x0s = all_cat_cpu(opt, log, pred_x0s)
        img_cond = all_cat_cpu(opt, log, img_cond)
        img_init_guess = all_cat_cpu(opt, log, init_guess)
        img_ground_truth = all_cat_cpu(opt, log, ground_truth)
        
        batch, len_t, *xdim = xs.shape
        
        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        log.info("Logging images ...")
        
        img_recon = xs[:, 0, ...]
        
        log_image("image/ground_truth", img_ground_truth)
        log_image("image/initial_guess", img_init_guess)
        log_image("image/condition", img_cond)
        log_image("image/reconstucted", img_recon)
        
        log_image("debug/gt_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj", xs.reshape(-1, *xdim), nrow=len_t)

        log.info(f"Evaluating metrices on {opt.val_batches} validation batches at NFE=1 ...")   
        
        # set deterministic sampling
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

            ground_truth, cond = batch
            init_guess, x1 = self.get_diffusion_endpoints(ground_truth, corrupt_method)
            init_guess, x1 = init_guess.to(opt.device), x1.to(opt.device)

            xs, pred_x0s = self.ddpm_sampling(val_opt, x1, init_guess, cond, nfe=1, verbose=False)
            
            recon = xs[:, 0, ...].to(val_opt.device)
            recon = all_cat_cpu(opt, log, recon)
            gt = all_cat_cpu(opt, log, ground_truth)

            avg_mae  += l1(recon, gt) * xs.shape[0]
            avg_mse  += l2(recon, gt) * xs.shape[0]
            avg_ssim += ssim(recon/2. + 0.5, gt/2. + 0.5) * xs.shape[0]

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


    @torch.no_grad()
    def evaluate(self, opt, log, val_dataset, corrupt_method):
        
        opt.cond_y = True 
        opt.ot_ode = True

        avg_mse  = 0.
        avg_mae  = 0.
        avg_ssim = 0.

        l1   = torch.nn.L1Loss(reduction='mean')
        l2   = torch.nn.MSELoss(reduction='mean')
        ssim = SSIM(window_size=11)

        val_loader = DataLoader(val_dataset,
            batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
        )
        pbar = tqdm(val_loader) 

        for batch in pbar:

            pbar.set_description(f'MAE - {avg_mae}, MSE - {avg_mse}, SSIM - {avg_ssim}')

            ground_truth, cond = batch
            init_guess, x1 = self.get_diffusion_endpoints(ground_truth, corrupt_method)
            init_guess, x1 = init_guess.to(opt.device), x1.to(opt.device)
            xs, _ = self.ddpm_sampling(opt, x1, init_guess, cond, nfe=opt.nfe, verbose=False)
            
            recon = xs[:, 0, ...].to(opt.device)
            ground_truth = ground_truth.to(opt.device)

            avg_mae  += l1(recon, ground_truth) * xs.shape[0] / len(val_dataset)
            avg_mse  += l2(recon, ground_truth) * xs.shape[0] / len(val_dataset)
            avg_ssim += ssim(recon/2. + 0.5, ground_truth/2. + 0.5) * xs.shape[0] / len(val_dataset)

        log.info(f'Average MAE on validation: {avg_mae}')
        log.info(f'Average MSE on validation: {avg_mse}')
        log.info(f'Average SSIM on validation: {avg_ssim}')