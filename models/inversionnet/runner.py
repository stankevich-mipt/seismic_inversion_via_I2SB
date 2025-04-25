# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as tu

from tqdm.auto import tqdm as tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .. import util
from utils.pytorch_ssim import SSIM
from models.runner import BaseRunner, build_optimizer_sched, all_cat_cpu, collect_all_subset
from .network import UNet


class Runner(BaseRunner):

    def __init__(self, opt, log, save_opt=True):
        super().__init__(opt, log, save_opt, model=UNet)

    def get_model_input(self, opt, c0, cond, corrupt_method):

        with torch.no_grad(): 
            c1 = corrupt_method(c0)

        cond = cond if self.cond else None
        if not (cond is None):
            input_ = torch.cat((c1, cond), dim=1)
        else:
            input_ = c1
            
        return  input_.to(opt.device)

    def train(
        self, opt, train_dataset, val_dataset, corrupt_method,
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

                batch = next(train_loader)  
                label, cond = batch[0].to(opt.device), batch[1].to(opt.device)           
                input_ = self.get_model_input(opt, label, cond, corrupt_method)
                pred = net(input_)

                assert pred.shape == label.shape

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
    def validate(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        ref_model, seismic_data = next(val_loader)
        input_ = self.get_model_input(opt, ref_model, seismic_data, corrupt_method)
        recon_model = self.net(input_)

        log.info("Collecting tensors ...")
        
        seismic_data = all_cat_cpu(opt, log, seismic_data)
        ref_model = all_cat_cpu(opt, log, ref_model) 
        smooth_model = all_cat_cpu(opt, log, input_[:, [0]])
        recon_model = all_cat_cpu(opt, log, recon_model)

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        log.info("Logging images ...")
        
        log_image("image/reference_model", ref_model)
        log_image("image/smooth_model", smooth_model)
        log_image("image/seismic_data", seismic_data[:, [2]])
        log_image("image/reconstucted_model", recon_model)

        log.info(f"Evaluating metrices on {opt.val_batches} validation batches")   

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
            input_ = self.get_model_input(opt, ref_model, seismic_data, corrupt_method)
            recon_model = self.net(input_)

            recon_model = all_cat_cpu(opt, log, recon_model)
            ref_model = all_cat_cpu(opt, log, ref_model)

            avg_mae  += l1(recon_model, ref_model) * ref_model.shape[0]
            avg_mse  += l2(recon_model, ref_model) * ref_model.shape[0]
            avg_ssim += ssim(recon_model/2 + 0.5, ref_model/2+ 0.5) * ref_model.shape[0]

            total_samples += ref_model.shape[0]
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
    def evaluate(self, opt, log, val_dataset, corrupt_method):\
    
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

            ref_model, seismic_data = batch[0].to(opt.device), batch[1].to(opt.device)
            input_ = self.get_model_input(opt, ref_model, seismic_data, corrupt_method)
            recon_model = self.net(input_)

            recon_model = all_cat_cpu(opt, log, recon_model)
            ref_model = all_cat_cpu(opt, log, ref_model)

            avg_mae  += l1(recon_model, ref_model) * ref_model.shape[0] / len(val_dataset)
            avg_mse  += l2(recon_model, ref_model) * ref_model.shape[0] / len(val_dataset)
            avg_ssim += ssim(recon_model/2 + 0.5, ref_model/2+ 0.5) * ref_model.shape[0] / len(val_dataset)

        log.info(f'Average MAE on validation: {avg_mae}')
        log.info(f'Average MSE on validation: {avg_mse}')
        log.info(f'Average SSIM on validation: {avg_ssim}')


    @torch.no_grad()
    def sample(self, opt, val_loader, corrupt_method):

        img_collected = 0

        val_loader = iter(val_loader)

        for loader_itr, _ in enumerate(val_loader):

            if loader_itr == opt.total_batches: break

            ref_model, smooth_model, seismic_data = self.get_data_triplet(opt, val_loader, corrupt_method)
            input_ = self.get_model_input(opt, ref_model, seismic_data, corrupt_method)
            recon_model = self.net(input_)
            
            gathered_ref_model = collect_all_subset(ref_model, log=None)
            gathered_seismic_data = collect_all_subset(seismic_data, log=None)
            gathered_smooth_model = collect_all_subset(smooth_model, log=None)
            gathered_recon_model = collect_all_subset(recon_model, log=None)

            if opt.global_rank == 0:

                save_path = opt.sample_dir / f"batch_{loader_itr}.pt"
                torch.save({
                    "ref_model": gathered_ref_model,
                    "seismic_data": gathered_seismic_data, 
                    "smooth_model": gathered_smooth_model,
                    "recon_model": gathered_recon_model,
                }, save_path)
        
            img_collected += len(gathered_recon_model)
            dist.barrier()

        return img_collected