from typing import Optional, List
from pydantic import StrictStr, StrictInt, StrictFloat, StrictBool
from tensorfn.config import (
    MainConfig,
    Config,
    Optimizer,
    Scheduler,
    DataLoader,
    Instance,
)

import diffusion
import model
from models.unet_autoenc import BeatGANsAutoencConfig


class Diffusion(Config):
    beta_schedule: Instance

class Dataset(Config):
    name: StrictStr
    path: StrictStr
    resolution: StrictInt

class Training(Config):
    ckpt_path: StrictStr
    optimizer: Optimizer
    scheduler: Optional[Scheduler]
    dataloader: DataLoader


class Eval(Config):
    wandb: StrictBool
    save_every: StrictInt
    valid_every: StrictInt
    log_every: StrictInt


class DiffusionConfig(MainConfig):
    diffusion: Diffusion
    training: Training


def get_model_conf():

    return BeatGANsAutoencConfig(image_size=(16,16), 
    in_channels=64*6, 
    model_channels=64*6, 
    out_channels=64*6*2,  # also learns sigma
    num_res_blocks=2, 
    num_input_res_blocks=None, 
    embed_channels=512, 
    attention_resolutions=(32,), 
    time_embed_channels=None, 
    dropout=0.1, 
    channel_mult=(1, 1, 1, 1, 1, 1), 
    input_channel_mult=None, 
    conv_resample=True, 
    dims=2, 
    num_classes=None, 
    use_checkpoint=False,
    num_heads=8, 
    num_head_channels=-1, 
    num_heads_upsample=-1, 
    resblock_updown=True, 
    use_new_attention_order=False, 
    resnet_two_cond=True, 
    resnet_cond_channels=None, 
    resnet_use_zero_module=True, 
    attn_checkpoint=False, 
    latent_net_conf=None)