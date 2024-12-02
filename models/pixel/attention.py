import torch 
from torch import nn
from einops import rearrange, repeat
from typing import Optional
from diffusers.models.attention import CrossAttention, FeedForward
import torch.nn.functional as F

class Transformer3DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 4,
        attention_head_dim: int = 64,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 4,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        hidden_states = self.proj_in(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                video_length=video_length
            )

        # Output
        hidden_states = (
            hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
        )
        hidden_states = self.proj_out(hidden_states)


        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output
    

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.1,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = False

        # SC-Attn
        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        self.attn_temp = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = nn.LayerNorm(dim)


    def forward(self, hidden_states, video_length=None):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states)
        )

        hidden_states = self.attn1(norm_hidden_states, video_length=video_length) + hidden_states

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        norm_hidden_states = (
            self.norm_temp(hidden_states)
        )
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class SparseCausalAttention(CrossAttention):
    def forward(self, hidden_states, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        encoder_hidden_states = hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        attention_mask = None

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        hidden_states = torch.matmul(attention_weights, value)

        hidden_states = rearrange(hidden_states, "(b f) d c -> b d (c f)", b=batch_size)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states