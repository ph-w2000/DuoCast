from typing import Union
from .unet import BeatGANsUNetModel, BeatGANsUNetConfig
from .unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel
Model = Union[BeatGANsUNetModel, BeatGANsAutoencModel]
ModelConfig = Union[BeatGANsUNetConfig, BeatGANsAutoencConfig]

from .unet_autoenc import BeatGANsAutoencConfig


def get_model(in_shape, T_in, T_out, **kwargs):
    C,H,W = in_shape

    pixel_model = BeatGANsAutoencConfig(image_size=(H,W), 
        in_channels=C*T_in, 
        model_channels=128, 
        out_channels=C*T_in,  # also learns sigma
        num_res_blocks=2, 
        num_input_res_blocks=None, 
        attention_resolutions=((16,16),(8,8),(4,4)), 
        time_embed_channels=None, 
        dropout=0.1, 
        channel_mult=(1, 1, 2, 2, 4, 4), 
        input_channel_mult=None, 
        conv_resample=True, 
        dims=2, 
        num_classes=None, 
        use_checkpoint=False,
        num_heads=1, 
        num_head_channels=-1, 
        num_heads_upsample=-1, 
        resblock_updown=True, 
        use_new_attention_order=False, 
        resnet_two_cond=True, 
        resnet_cond_channels=None, 
        resnet_use_zero_module=True, 
        attn_checkpoint=False, 
        latent_net_conf=None
        ).make_model()

    return pixel_model