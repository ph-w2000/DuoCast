from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu

from .latentnet import *
from .unet import *
from .choices import *
from .resnet import ResNet
from einops import rearrange

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        self.encoder = ResNet(conf.model_channels, resnet_type='18')


        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x):

        cond = self.encoder(x)

        return {'cond': cond}

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S


    def forward_with_cond_scale(
        self,
        x,
        t,
        cond,
        cond_scale,
    ):
        logits = self.forward(x, t, cond=cond, prob = 1)

        null_logits = self.forward(x, t, cond=cond, prob = 0)

        return [null_logits + (logits - null_logits) * 7.5, logits, null_logits]

    def forward(self,
                x,
                t,
                x_cond=None,
                prob=1,
                y=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """
        cond_mask = prob_mask_like((x.shape[0],), prob = prob, device = x.device)

        # if t_cond is None:
        #     t_cond = t

        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)

        if cond is None:
            x_cond  = (cond_mask.view(-1,1,1,1,1)*x_cond)
            if x is not None:
                assert len(x) == len(x_cond), f'{len(x)} != {len(x_cond)}'
            tmp = self.encode(x_cond)
            cond = tmp['cond']
        else:
            if prob==1:
                cond = cond[0]
            elif prob==0:
                cond = cond[1]
            

        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            # _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            # _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                # time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style

        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond
        mid_cond_emb = cond[-1]
        dec_cond_emb = cond #+ [cond[-1]]

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]


        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    if k <= 11 and k>=9:
                        h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb[0])
                    elif k <= 14 and k>=12:
                        h = self.input_blocks[k](h,
                                                emb=enc_time_emb,
                                                cond=enc_cond_emb[1])
                    elif k <= 17 and k>=15:
                        h = self.input_blocks[k](h,
                                                emb=enc_time_emb,
                                                cond=enc_cond_emb[2])
                    else:
                        h = self.input_blocks[k](h,
                                            emb=enc_time_emb,
                                            cond=None)

                    hs[i].append(h)
                    #h = th.concat([h, enc_cond_emb[k]], 1)

                    k += 1

            assert k == len(self.input_blocks)
            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]
        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)
                if k <= 2 and k>=0:
                    h = self.output_blocks[k](h,
                                        emb=dec_time_emb,
                                        cond=dec_cond_emb[2],
                                        lateral=lateral)
                elif k <= 5 and k>=3:
                    h = self.output_blocks[k](h,
                                        emb=dec_time_emb,
                                        cond=dec_cond_emb[1],
                                        lateral=lateral)
                elif k <= 8 and k>=6:
                    h = self.output_blocks[k](h,
                                        emb=dec_time_emb,
                                        cond=dec_cond_emb[0],
                                        lateral=lateral)
                else:
                    h = self.output_blocks[k](h,
                                            emb=dec_time_emb,
                                            cond=None,
                                            lateral=lateral)
                k += 1
        pred = self.out(h)
        return pred


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)
