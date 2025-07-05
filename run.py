import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp
import math
import time
import argparse
import logging 
import yaml
import cProfile
from tqdm import tqdm
from datetime import timedelta
import wandb

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs
from ema_pytorch import EMA
from diffusers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from accelerate.utils import gather_object

from datasets.get_datasets import get_dataset
from utils.metrics import Evaluator
from utils.tools import print_log, cycle, show_img_info
from models.vae.autoencoder_kl import AutoencoderKL

os.environ["ACCELERATE_DEBUG_MODE"] = "1"

def create_parser():
    # --------------- Basic ---------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone', default='simvp',  type=str,                 help='backbone model for deterministic prediction')
    parser.add_argument('--use_diff', action="store_true", default=True,        help='Weather use diff framework, as for ablation study')
    
    parser.add_argument("--seed",           type=int,   default=0,              help='Experiment seed')
    parser.add_argument("--exp_dir",        type=str,   default='basic_exps',   help="experiment directory")
    parser.add_argument("--exp_note",       type=str,   default=None,           help="additional note for experiment")


    # --------------- Dataset ---------------
    parser.add_argument("--dataset",        type=str,   default='sevir',        help="dataset name")
    parser.add_argument("--img_size",       type=int,   default=128,            help="image size")
    parser.add_argument("--img_channel",    type=int,   default=1,              help="channel of image")
    parser.add_argument("--seq_len",        type=int,   default=25,             help="sequence length sampled from dataset")
    parser.add_argument("--frames_in",      type=int,   default=5,              help="number of frames to input")
    parser.add_argument("--frames_out",     type=int,   default=20,             help="number of frames to output")    
    parser.add_argument("--num_workers",    type=int,   default=8,              help="number of workers for data loader")
    
    # --------------- Optimizer ---------------
    parser.add_argument("--lr",             type=float, default=1e-4,            help="learning rate")
    parser.add_argument("--lr-beta1",       type=float, default=0.90,            help="learning rate beta 1")
    parser.add_argument("--lr-beta2",       type=float, default=0.95,            help="learning rate beta 2")
    parser.add_argument("--l2-norm",        type=float, default=0.0,             help="l2 norm weight decay")
    parser.add_argument("--ema_rate",       type=float, default=0.95,            help="exponential moving average rate")
    parser.add_argument("--scheduler",      type=str,   default='constant',        help="learning rate scheduler", choices=['constant', 'linear', 'cosine'])
    parser.add_argument("--warmup_steps",   type=int,   default=1000,            help="warmup steps")
    parser.add_argument("--mixed_precision",type=str,   default='no',            help="mixed precision training")
    parser.add_argument("--grad_acc_step",  type=int,   default=1,               help="gradient accumulation step")
    
    # --------------- Training ---------------
    parser.add_argument("--batch_size",     type=int,   default=8,              help="batch size")
    parser.add_argument("--epochs",         type=int,   default=40,              help="number of epochs")
    parser.add_argument("--training_steps", type=int,   default=200000,          help="number of training steps")
    parser.add_argument("--early_stop",     type=int,   default=10,              help="early stopping steps")
    parser.add_argument("--ckpt_milestone", type=str,   default=None,            help="resumed checkpoint milestone")
    
    # --------------- Additional Ablation Configs ---------------
    parser.add_argument("--eval",           action="store_true",                 help="evaluation mode")
    parser.add_argument("--wandb_state", type=str, default='offline', help="wandb state config")
    parser.add_argument("--continue_train",           action="store_true",)

    args = parser.parse_args()
    return args


class Runner(object):
    
    def __init__(self, args):
        
        self.args = args
        self._preparation()
        
        self.accelerator = Accelerator()
        
        print_log('============================================================', self.is_main)
        print_log("                 Experiment Start                           ", self.is_main)
        print_log('============================================================', self.is_main)
    
        print_log(self.accelerator.state, self.is_main)
        
        self._load_data()
        self._build_model()
        self._build_optimizer()
        
        # distributed ema for parallel sampling

        self.model, self.vae, self.optimizer,  self.scheduler, self.train_loader, self.valid_loader, self.test_loader = self.accelerator.prepare(
            self.model, self.vae,
            self.optimizer, self.scheduler,
            self.train_loader, self.valid_loader, self.test_loader
        )

        print_log(f"gpu_nums: {torch.cuda.device_count()}, gpu_id: {torch.cuda.current_device()}")
        
        # if self.args.ckpt_milestone is not None:
        #     self.load(self.args.ckpt_milestone)

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def device(self):
        return self.accelerator.device
    
    def _preparation(self):
        # =================================
        # Build Exp dirs and logging file
        # =================================

        set_seed(self.args.seed)
        self.model_name = self.model_name = ('DuoCast' if self.args.use_diff else 'Single') + self.args.backbone
        self.exp_name   = f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}"
        
        cur_dir         = os.path.dirname(os.path.abspath(__file__))
        
        self.exp_dir    = osp.join(cur_dir, 'Exps', self.args.exp_dir, self.exp_name)        
        self.ckpt_path  = osp.join(self.exp_dir, 'checkpoints')
        self.valid_path = osp.join(self.exp_dir, 'valid_samples')
        self.test_path  = osp.join(self.exp_dir, 'test_samples')
        self.log_path   = osp.join(self.exp_dir, 'logs')
        self.sanity_path = osp.join(self.exp_dir, 'sanity_check')
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        exp_params      = self.args.__dict__
        params_path     = osp.join(self.exp_dir, 'params.yaml')
        yaml.dump(exp_params, open(params_path, 'w'))
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            # filemode='a',
            handlers=[
                logging.FileHandler(osp.join(self.log_path, 'log.log')),
                # logging.StreamHandler()
            ]
        )
        
    def _load_data(self):
        # =================================
        # Get Train/Valid/Test dataloader among datasets 
        # =================================

        train_data, valid_data, test_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=self.args.dataset,
            # data_path=self.args.data_path,
            img_size=self.args.img_size,
            seq_len=self.args.seq_len,
            batch_size=self.args.batch_size,
        )
        
        self.visiual_save_fn = color_save_fn
        self.thresholds      = THRESHOLDS
        self.scale_value     = PIXEL_SCALE
        
        if self.args.dataset != 'sevir':
            # preload big batch data for gradient accumulation
            self.train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size*self.args.grad_acc_step, shuffle=True, num_workers=self.args.num_workers, drop_last=True
            )
            self.valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.args.batch_size , shuffle=False, num_workers=self.args.num_workers
            )
        else:
            self.train_loader = train_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.valid_loader = valid_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)
            
            
        print_log(f"train data: {len(self.train_loader)}, valid data: {len(self.valid_loader)}, test_data: {len(self.test_loader)}",
                  self.is_main)
        print_log(f"Pixel Scale: {PIXEL_SCALE}, Threshold: {str(THRESHOLDS)}",
                  self.is_main)
        
    def _build_model(self):
        # =================================
        # import and create different models given model config
        # =================================

        if self.args.backbone == 'simvp':
            from models.simvp import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
            }
            model = get_model(**kwargs)
        
        elif self.args.backbone == 'phydnet':
            from models.phydnet import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "device": self.device
            }
            model = get_model(**kwargs)
        
        elif self.args.backbone == 'pixel':
            from models.pixel import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "device": self.device
            }
            model = get_model(**kwargs)
            
        else:
            raise NotImplementedError
        
        if self.args.use_diff:
            from duocast import get_model
            kwargs = {
                'img_channels' : self.args.img_channel,
                'dim' : 64,
                'dim_mults' : (1,2,4,8),
                'T_in': self.args.frames_in,
                'T_out': self.args.frames_out,
                'sampling_timesteps': 5,
            }
            diff_model = get_model(**kwargs)
            diff_model.load_backbone(model)
            model = diff_model
            
        self.model = model
        self.ema = EMA(self.model, beta=self.args.ema_rate, update_every=20).to(self.device)        
        
        if self.is_main:
            total = sum([param.nelement() for param in self.model.parameters()])
            print_log("Main Model Parameters: %.2fM" % (total/1e6), self.is_main)

        self.vae = AutoencoderKL(
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            in_channels=1,
            block_out_channels=[128, 256, 512, 512],
            act_fn='silu',
            latent_channels=64,
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            norm_num_groups=32,
            layers_per_block=2,
            out_channels=1).eval()

        state_dict = torch.load('pretrained_sevirlr_vae_8x8x64_v1.pt',map_location=torch.device("cpu"))
        self.vae.load_state_dict(state_dict=state_dict)


    def _build_optimizer(self):
        # =================================
        # Calcutate training nums and config optimizer and learning schedule
        # =================================
        num_steps_per_epoch = len(self.train_loader)
        num_epoch = math.ceil(self.args.training_steps / num_steps_per_epoch)
        self.global_epochs = self.args.epochs
        self.global_steps = self.global_epochs * num_steps_per_epoch
        self.steps_per_epoch = math.ceil(num_steps_per_epoch / self.accelerator.num_processes)
        
        self.cur_step, self.cur_epoch = 0, 0

        warmup_steps = self.args.warmup_steps

        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            betas=(self.args.lr_beta1, self.args.lr_beta2),
            weight_decay=self.args.l2_norm
        )
        if self.args.scheduler == 'constant':
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        elif self.args.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=self.global_steps,
            )
        elif self.args.scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps , 
                num_training_steps=self.global_steps,
            )
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}".format(
                    self.args.scheduler
            )
        )
            
        if self.is_main:
            print_log("============ Running training ============")
            print_log(f"    Num examples = {len(self.train_loader)}")
            print_log(f"    Num Epochs = {self.global_epochs}")
            print_log(f"    Instantaneous batch size per GPU = {self.args.batch_size}")
            print_log(f"    Total train batch size (w. parallel, distributed & accumulation) = {self.args.batch_size * self.accelerator.num_processes}")
            print_log(f"    Total optimization steps = {self.global_steps}")
            print_log(f"optimizer: {self.optimizer} with init lr: {self.args.lr}")
        
    
    def save(self):
        # =================================
        # Save checkpoint state for model and ema
        # =================================
        if not self.is_main:
            return
        
        data = {
            'step': self.cur_step,
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'ema': self.ema.state_dict(),
            'opt': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),

        }
        
        torch.save(data, osp.join(self.ckpt_path, f"ckpt-{self.cur_step}.pt"))
        print_log(f"Save checkpoint {self.cur_step} to {self.ckpt_path}", self.is_main)
        
        
    def load(self, milestone, continue_training=False):
        # =================================
        # load model checkpoint
        # =================================        
        device = self.accelerator.device
        
        if '.pt' in milestone:
            data = torch.load(milestone, map_location=device)
        else:
            data = torch.load(osp.join(self.ckpt_path, f"ckpt-{milestone}.pt"), map_location=device)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'],strict=False)
        self.model = self.accelerator.prepare(model)
        
        if continue_training:
            self.optimizer.load_state_dict(data['opt'])
            self.scheduler.load_state_dict(data['scheduler'])
        
        if self.is_main:
            self.ema.load_state_dict(data['ema'],strict=False)

        # self.cur_epoch = data['epoch']
        # self.cur_step = data['step']
        print_log(f"Load checkpoint {milestone} from {self.ckpt_path}", self.is_main)
        
    
    def train(self):
        if self.accelerator.is_main_process:
            # os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = self.exp_dir 
            wandb.init(project="DuoCast", name="latent", config={})
            wandb.config.update(self.args) 

        # set global step as traing process
        tqdm_total = math.ceil(self.global_steps / self.accelerator.num_processes)
        pbar = tqdm(
            initial=self.cur_step,
            total=tqdm_total,
            disable=not self.is_main,
        )
        start_epoch = self.cur_epoch
        for epoch in range(start_epoch, self.global_epochs):
            self.cur_epoch = epoch
            self.model.train()
            
            for i, batch in enumerate(self.train_loader):
                # train the model with mixed_precision
                with self.accelerator.autocast(self.model):

                    loss_dict = self._train_batch(batch)
                    self.accelerator.backward(loss_dict['total_loss'])
                    
                    if self.cur_step == 0:
                        # training process check
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                print_log(name, self.is_main)
    
                self.accelerator.wait_for_everyone()
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step()
                
                # record train info
                lr = self.optimizer.param_groups[0]['lr']
                log_dict = dict()
                log_dict['lr'] = lr
                for k,v in loss_dict.items():
                    log_dict[k] = v.item()
                self.accelerator.log(log_dict, step=self.cur_step)
                pbar.set_postfix(**log_dict)   
                state_str = f"Epoch {self.cur_epoch}/{self.global_epochs}, Step {i}/{self.steps_per_epoch}"
                pbar.set_description(state_str)
                
                # update ema param and log file every 20 steps
                if i % 20 == 0:
                    logging.info(state_str+'::'+str(log_dict))
                self.ema.update()

                if self.accelerator.is_main_process:
                    wandb.log(
                        {
                            "Loss/total_loss": loss_dict['total_loss'].item(),
                            "Loss/phydnet_loss": loss_dict['phydnet'].item(),
                            "Loss/pixel_ep_loss": loss_dict['pixel_ep'].item(),
                            "Loss/latent_ep_loss": loss_dict['latent_ep'].item(),
                            "LR": lr,
                        }, 
                        step=self.cur_step
                    )

                self.cur_step += 1
                pbar.update(1)
                
            # save checkpoint and do test every epoch

            if epoch >15:
                self.save()
            print_log(f" ========= Finisth one Epoch ==========", self.is_main)

        self.accelerator.end_training()
        
    def _get_seq_data(self, batch):
        # frame_seq = batch['vil'].unsqueeze(2).to(self.device)
        return batch      # [B, T, C, H, W]
    
    def _train_batch(self, batch):
        radar_batch = self._get_seq_data(batch)
        frames_in, frames_out = radar_batch[:,:self.args.frames_in], radar_batch[:,self.args.frames_in:]
        assert radar_batch.shape[1] == self.args.frames_out + self.args.frames_in, "radar sequence length error"
        loss = self.model(
            frames_in=frames_in.to(self.device, non_blocking=True), 
            frames_gt=frames_out.to(self.device, non_blocking=True),
            compute_loss=True,
            vae=self.vae
        )
        if loss is None:
            raise ValueError("Loss is None, please check the model predict function")
        loss['total_loss'] = loss['phydnet'] * 0.1 + loss['pixel_ep'] * 0.1 + loss['latent_ep'] * 0.5
        return loss
        
    
    @torch.no_grad()
    def _sample_batch(self, batch, use_ema=False):
        
        sample_fn = self.ema.ema_model.predict if use_ema else self.model.predict
        frame_in = self.args.frames_in
        radar_batch = self._get_seq_data(batch)
        radar_input, radar_gt = radar_batch[:,:frame_in], radar_batch[:,frame_in:]
        radar_pred, _ = sample_fn(radar_input, self.vae, compute_loss=False)
        
        radar_gt = self.accelerator.gather(radar_gt).detach().cpu().numpy()
        radar_pred = self.accelerator.gather(radar_pred).detach().cpu().numpy()

        return radar_gt, radar_pred
    
    
    def test_samples(self, milestone, do_test=False):
        # init test data loader
        data_loader = self.test_loader if do_test else self.valid_loader
        # init sampling method
        self.model.eval()
        # init test dir config
        cnt = 0
        save_dir = osp.join(self.test_path, f"sample-{milestone}") if do_test else osp.join(self.valid_path, f"sample-{milestone}")
        os.makedirs(save_dir, exist_ok=True)

        if self.is_main:
            eval = Evaluator(
                seq_len=self.args.frames_out,
                value_scale=self.scale_value,
                thresholds=self.thresholds,
                save_path=save_dir,
            )

        # start test loop
        sample_fn = self.model.predict
        cnt = 0
        for batch in tqdm(data_loader,desc='Test Samples', disable=not self.is_main):
            result_list = []
            # sample
            frame_in = self.args.frames_in
            radar_batch = self._get_seq_data(batch)
            radar_input, radar_gt = radar_batch[:,:frame_in], radar_batch[:,frame_in:]
            radar_pred, _ = sample_fn(radar_input.to(self.device, non_blocking=True), self.vae, compute_loss=False )
            result_list.append([radar_gt, radar_pred])

            result_list = gather_object(result_list)
            if self.is_main:
                for radar_ori, radar_recon in result_list:
                    eval.evaluate(radar_ori, radar_recon)
                if cnt >= 0:
                    for i in range(radar_input.shape[0]):
                        self.visiual_save_fn(radar_input[i], radar_pred[i], radar_gt[i], osp.join(save_dir, f"{cnt}-{i}"), data_type='vil')
            cnt+=1
            self.accelerator.wait_for_everyone()
        
        if self.is_main:
            # test done
            res = eval.done()
            print_log(f"Test Results: {res}")
            print_log("="*30)

        
    def check_milestones(self, target_ckpt=None):
        self.load(target_ckpt)
        saved_dir_name = target_ckpt.split('/')[-1].split('.')[0]
        self.test_samples(saved_dir_name, do_test=True)
            
def main():
    args = create_parser()
    exp = Runner(args)
    if not args.eval:
        exp.load(args.ckpt_milestone, continue_training=args.continue_train)
        exp.train()
    else:
        exp.check_milestones(target_ckpt=args.ckpt_milestone)
    

if __name__ == '__main__':
    main()