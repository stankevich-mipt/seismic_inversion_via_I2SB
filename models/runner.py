import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tu
import torch.distributed as dist
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

    opt.start_itr = 0

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "step" in checkpoint.keys():
            opt.start_itr = checkpoint["step"]
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

    log.info(f"[Opt] Running optimization from step {opt.start_itr}")

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


def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)


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
  
    def get_data_triplet(self, opt, dataloader, corrupt_method):

        ref_model, seismic_data = next(dataloader)
        smooth_model = corrupt_method(ref_model)
        
        ref_model = ref_model.to(opt.device) 
        smooth_model = smooth_model.to(opt.device)
        seismic_data = seismic_data.to(opt.device)

        return ref_model, smooth_model, seismic_data
    
    def augment_batch(self, c0, c1, cond):
        """
        Augmentation in the base class does nothing
        """
        return c0, c1, cond

    def compute_label(self, step, c0, ct, pred_c0=False):
        """
        Framework-dependent regression label computation
        """
        raise NotImplementedError

    def compute_pred_c0(self, step, ct, net_out, pred_c0=False, clip_denoise=False):
        """
        Framework-dependent handling of neural net prediction
        """
        return NotImplementedError

    def prepare_training_signature(self, opt, dataloader, corrupt_method):
        """
        Method that aggregates all the nesessary manipulations  
        with model inputs and outputs during training. 
        Signature is framework-specific and should be defined separately.   
        """
        raise NotImplementedError

    def train(
        self, opt, train_dataset, val_dataset, corrupt_method
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
        
        for it in range(opt.start_itr, opt.num_itr):
            
            optimizer.zero_grad()

            for _ in range(n_inner_loop):            

                step, ct, cond, label, weights = self.prepare_training_signature(
                    opt, train_loader, corrupt_method 
                )

                pred = net(ct, step, cond=cond)

                assert label.shape == pred.shape

                loss = F.mse_loss(pred, label, weight=weights)
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
            
            if it % opt.loss_log_freq == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % opt.save_freq == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "step": it,
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / f"ckpt{it}.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it % opt.val_freq == 0:
                net.eval()
                self.validate(opt, it, val_loader, corrupt_method)
                net.train()
        
        self.writer.close()
        
    @torch.no_grad()
    def ddpm_sampling(
        self, opt, smooth_model, seismic_data,
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

        
        # seismic data is not required if model is unconditional 
        seismic_data = seismic_data if self.cond else None
        
        with self.ema.average_parameters():
            
            self.net.eval()

            def pred_c0_fn(ct, step):

                step = torch.full((ct.shape[0],), step, device=opt.device, dtype=torch.long)
                
                guidance_scale = getattr(opt, 'guidance_scale', None)
                
                # If guidance scale is present, apply formula 6 from https://arxiv.org/abs/2207.12598
                if guidance_scale is None:
                    out = self.net(ct, step, cond=seismic_data)
                else:
                    out_cond   = self.net(ct, step, seismic_data)
                    out_uncond = self.net(ct, step, 0.*seismic_data)
                    out = guidance_scale * out_cond + (1 - guidance_scale) * out_uncond

                return self.compute_pred_c0(step, ct, out, pred_c0=opt.pred_c0, clip_denoise=opt.clip_denoise)

            traj_ct, traj_c0 = self.diffusion.ddpm_sampling(
                steps, pred_c0_fn, smooth_model, log_steps=log_steps, verbose=verbose, ot_ode=opt.ot_ode
            )

        b, *xdim = smooth_model.shape
        assert traj_ct.shape == traj_c0.shape == (b, log_count, *xdim)

        return traj_ct, traj_c0

    @torch.no_grad()
    def validate(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        ref_model, smooth_model, seismic_data = self.get_data_triplet(opt, val_loader, corrupt_method)
        
        ct_traj, c0_traj = self.ddpm_sampling(opt, smooth_model, seismic_data, log_count=10)
        log.info(f"Generated recon trajectories: size={ct_traj.shape}")

        # pick central shot for display
        seismic_data = seismic_data[:, [2]]

        log.info("Collecting tensors ...")
        
        ct_traj = all_cat_cpu(opt, log, ct_traj)
        c0_traj = all_cat_cpu(opt, log, c0_traj)
        seismic_data = all_cat_cpu(opt, log, seismic_data)
        ref_model = all_cat_cpu(opt, log, ref_model)
        smooth_model = all_cat_cpu(opt, log, smooth_model)

        batch, len_t, *xdim = ct_traj.shape
        
        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        log.info("Logging images ...")
        
        recon_model = ct_traj[:, 0, ...]
        
        log_image("inference/c0_traj", c0_traj.reshape(-1, *xdim), nrow=len_t)
        log_image("inference/ct_traj", ct_traj.reshape(-1, *xdim), nrow=len_t)

        log_image("image/reference_model", ref_model)
        log_image("image/smooth_model", smooth_model)
        log_image("image/seismic_data", seismic_data)
        log_image("image/reconstucted_model", recon_model)
        
        log.info(f"Evaluating metrices on {opt.val_batches} validation batches at NFE=1 ...")   
        
        avg_mse  = 0.
        avg_mae  = 0.
        avg_ssim = 0.
        total_samples = 0.

        l1   = torch.nn.L1Loss(reduction='mean')
        l2   = torch.nn.MSELoss(reduction='mean')
        ssim = SSIM(window_size=11)

        processed_batches = int(0)

        for batch in val_loader:

            if processed_batches == opt.val_batches: break

            ref_model, seismic_data = batch[0].to(opt.device), batch[1].to(opt.device)
            smooth_model = corrupt_method(ref_model)
            ct_traj, c0_traj = self.ddpm_sampling(opt, smooth_model, seismic_data, nfe=1, verbose=False)
            recon_model = ct_traj[:, 0, ...].to(opt.device)
            
            recon_model = all_cat_cpu(opt, log, recon_model)
            ref_model = all_cat_cpu(opt, log, ref_model)

            avg_mae  += l1(recon_model, ref_model) * ct_traj.shape[0]
            avg_mse  += l2(recon_model, ref_model) * ct_traj.shape[0]
            avg_ssim += ssim(recon_model/2 + 0.5, ref_model/2+ 0.5) * ct_traj.shape[0]

            total_samples += ct_traj.shape[0]
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

            ref_model, seismic_data = batch[0].to(opt.device), batch[1].to(opt.device)
            smooth_model = corrupt_method(ref_model)
            ct_traj, _ = self.ddpm_sampling(opt, smooth_model, seismic_data, nfe=opt.nfe, verbose=False)
            recon_model = ct_traj[:, 0, ...].to(opt.device)

            avg_mae  += l1(recon_model, ref_model) * ct_traj.shape[0] / len(val_dataset)
            avg_mse  += l2(recon_model, ref_model) * ct_traj.shape[0] / len(val_dataset)
            avg_ssim += ssim(recon_model/2. + 0.5, ref_model/2. + 0.5) * ct_traj.shape[0] / len(val_dataset)

        log.info(f'Average MAE on validation: {avg_mae}')
        log.info(f'Average MSE on validation: {avg_mse}')
        log.info(f'Average SSIM on validation: {avg_ssim}')

    @torch.no_grad()
    def sample(self, opt, val_loader, corrupt_method):

        img_collected = 0

        val_loader = iter(val_loader)

        for loader_itr, _ in enumerate(iter(val_loader)):

            if loader_itr == opt.total_batches: break

            ref_model, smooth_model, seismic_data = self.get_data_triplet(opt, val_loader, corrupt_method)
            ct_traj, c0_traj = self.ddpm_sampling(opt, smooth_model, seismic_data, nfe=opt.nfe, verbose=False)
            recon_model = ct_traj[:, 0, ...].contiguous()
            
            gathered_ref_model = collect_all_subset(ref_model, log=None)
            gathered_seismic_data = collect_all_subset(seismic_data, log=None)
            gathered_smooth_model = collect_all_subset(smooth_model, log=None)
            gathered_recon_model = collect_all_subset(recon_model, log=None)
            gathered_ct_traj = collect_all_subset(ct_traj, log=None)
            gathered_c0_traj = collect_all_subset(c0_traj, log=None)

            if opt.global_rank == 0:

                save_path = opt.sample_dir / f"batch_{loader_itr}.pt"
                torch.save({
                    "ref_model": gathered_ref_model,
                    "seismic_data": gathered_seismic_data, 
                    "smooth_model": gathered_smooth_model,
                    "recon_model": gathered_recon_model,
                    "ct_traj": gathered_ct_traj,
                    "c0_traj": gathered_c0_traj
                }, save_path)
        
            img_collected += len(gathered_recon_model)
            dist.barrier()

        return img_collected