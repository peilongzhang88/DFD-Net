import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
from torch import Tensor
from typing import Tuple
import numbers
from timm.models.layers import to_2tuple, trunc_normal_
from einops import rearrange
import gc
from collections import OrderedDict
from einops.layers.torch import Rearrange
from timm.models import register_model
from timm.models.layers import DropPath
from timm.models.vision_transformer import _cfg
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import Bottleneck, C2f, C3k

class LayerNorm2d(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.ln = nn.LayerNorm(channels)

  def forward(self, x):
    x = rearrange(x, "N C H W -> N H W C")
    x = self.ln(x)
    x = rearrange(x, "N H W C -> N C H W")
    return x

def init_linear(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

class Partial:
    def __init__(self, module, *args, **kwargs):
        self.module = module
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args_c, **kwargs_c):
        return self.module(*args_c, *self.args, **kwargs_c, **self.kwargs)

class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return x

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')
    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
class ConvFFN(nn.Module):
    def __init__(self, dim=768):
        super(ConvFFN, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x

class TopkRouting(nn.Module):
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)
        return r_weight, topk_index

class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               )
        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv
        return topk_kv

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        return q, kv

class QKVConv(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Conv2d(dim,  qk_dim + qk_dim + dim, 1, 1, 0)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=1)
        return q, kv
        
class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        self.scale = qk_scale or self.qk_dim ** -0.5
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        mul_weight = 'soft' if self.soft_routing else ('hard' if self.diff_routing else 'none')
        self.kv_gather = KVGather(mul_weight=mul_weight)
        if self.param_attention in ['qkvo', 'qkv']:
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim) if self.param_attention == 'qkvo' else nn.Identity()
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        if self.kv_downsample_mode == 'ada_avgpool':
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        else:
            self.kv_down = nn.Identity()
        self.attn_act = nn.Softmax(dim=-1)
        self.auto_pad=auto_pad

    def forward(self, x, ret_attn_mask=False):
        if self.auto_pad:
            N, H_in, W_in, C = x.size()
            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.size()
        else:
            N, H, W, C = x.size()
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)
        q, kv = self.qkv(x)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)
        r_weight, r_idx = self.router(q_win, k_win)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads)
        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win, h=H//self.n_win, w=W//self.n_win)
        out = out + lepe
        out = self.wo(out)
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()
        return out

class TransformerMLPWithConv(nn.Module):
    def __init__(self, channels, expansion, drop):
        super().__init__()
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(nn.Conv2d(self.dim1, self.dim2, 1, 1, 0))
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Sequential(nn.Conv2d(self.dim2, self.dim1, 1, 1, 0))
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

class DeBiLevelRoutingAttentionblcok(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=True, param_size='small'):
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        if param_size=='tiny' or param_size=='small':
            configs = {64: (1, 16, 9, 8, 3, 56), 128: (2, 16, 7, 4, 3, 28), 256: (4, 4, 5, 2, 3, 14), 512: (8, 49, 3, 1, 3 if param_size=='tiny' else 1, 7)}
            self.n_groups, self.top_k_def, self.kk, self.stride_def, self.expain_ratio, q_size_val = configs[self.dim]
            self.q_size = to_2tuple(q_size_val)
        elif param_size=='base':
            configs = {96: (1, 16, 9, 8, 3, 56), 192: (2, 16, 7, 4, 3, 28), 384: (3, 4, 5, 2, 3, 14), 768: (6, 49, 3, 1, 3, 7)}
            self.n_groups, self.top_k_def, self.kk, self.stride_def, self.expain_ratio, q_size_val = configs[self.dim]
            self.q_size = to_2tuple(q_size_val)
        self.q_h, self.q_w = self.q_size
        self.kv_h, self.kv_w = self.q_h // self.stride_def, self.q_w // self.stride_def
        self.n_group_channels = self.dim // self.n_groups
        self.n_group_heads = self.num_heads // self.n_groups
        self.offset_range_factor = -1
        self.head_channels = dim // num_heads
        self.scale = qk_scale or self.qk_dim ** -0.5
        self.rpe_table = nn.Parameter(torch.zeros(self.num_heads, self.q_h * 2 - 1, self.q_w * 2 - 1))
        trunc_normal_(self.rpe_table, std=0.01)
        self.lepe1 = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=self.stride_def, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else lambda x: torch.zeros_like(x)
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        self.router = TopkRouting(qk_dim=self.qk_dim, qk_scale=self.scale, topk=self.topk, diff_routing=self.diff_routing, param_routing=self.param_routing)
        mul_weight = 'soft' if self.soft_routing else ('hard' if self.diff_routing else 'none')
        self.kv_gather = KVGather(mul_weight=mul_weight)
        self.qkv_conv = QKVConv(self.dim, self.qk_dim)
        if self.kv_downsample_mode == 'ada_avgpool': self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool': self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool': self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool': self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        else: self.kv_down = nn.Identity()
        self.attn_act = nn.Softmax(dim=-1)
        self.auto_pad=auto_pad
        self.proj_q = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_k = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_v = nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.unifyheads1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.conv_offset_q = nn.Sequential(nn.Conv2d(self.n_group_channels, self.n_group_channels, self.kk, self.stride_def, self.kk//2, groups=self.n_group_channels, bias=False), LayerNormProxy(self.n_group_channels), nn.GELU(), nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = TransformerMLPWithConv(dim, self.expain_ratio, 0.)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device))
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        return ref[None, ...].expand(B * self.n_groups, -1, -1, -1)

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(torch.arange(0, H, dtype=dtype, device=device), torch.arange(0, W, dtype=dtype, device=device), indexing='ij')
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        return ref[None, ...].expand(B * self.n_groups, -1, -1, -1)

    def forward(self, x, ret_attn_mask=False):
        dtype, device = x.dtype, x.device
        if self.auto_pad:
            N, H_in, W_in, C = x.size()
            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.size()
        else:
            N, H, W, C = x.size()
        x_res = rearrange(x, "n h w c -> n c h w")
        q,kv = self.qkv_conv(x.permute(0, 3, 1, 2))
        q_bi = rearrange(q, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)
        kv = rearrange(kv, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)
        q_pix = rearrange(q_bi, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
        lepe1 = self.lepe1(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        q_off = rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset_q = self.conv_offset_q(q_off).contiguous()
        Hk, Wk = offset_q.size(2), offset_q.size(3)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset_q = offset_q.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset_q = rearrange(offset_q, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, N, dtype, device)
        pos_k = offset_q + reference if self.offset_range_factor >= 0 else (offset_q + reference).clamp(-1., +1.)
        x_sampled_q = F.grid_sample(input=x_res.reshape(N * self.n_groups, self.n_group_channels, H, W), grid=pos_k[..., (1, 0)], mode='bilinear', align_corners=True)
        q_sampled = x_sampled_q.reshape(N, C, Hk, Wk)
        if self.auto_pad:
            q_sampled=q_sampled.permute(0, 2, 3, 1)
            Ng, Hg, Wg, Cg = q_sampled.size()
            pad_l = pad_t = 0
            pad_rg = (self.n_win - Wg % self.n_win) % self.n_win
            pad_bg = (self.n_win - Hg % self.n_win) % self.n_win
            q_sampled = F.pad(q_sampled, (0, 0, pad_l, pad_rg, pad_t, pad_bg))
            _, Hg, Wg, _ = q_sampled.size()
            q_sampled=q_sampled.permute(0, 3, 1, 2)
            lepe1 = F.pad(lepe1.permute(0, 2, 3, 1), (0, 0, pad_l, pad_rg, pad_t, pad_bg))
            lepe1=lepe1.permute(0, 3, 1, 2)
            pos_k = F.pad(pos_k, (0, 0, pad_l, pad_rg, pad_t, pad_bg))
        queries_def = self.proj_q(q_sampled)
        queries_def = rearrange(queries_def, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win).contiguous()
        q_win, k_win = queries_def.mean([2, 3]), kv[..., 0:(self.qk_dim)].mean([2, 3])
        r_weight, r_idx = self.router(q_win, k_win)
        kv_gather = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_gather, v_gather = kv_gather.split([self.qk_dim, self.dim], dim=-1)
        k = rearrange(k_gather, 'n p2 k hw (m c) -> (n p2) m c (k hw)', m=self.num_heads)
        v = rearrange(v_gather, 'n p2 k hw (m c) -> (n p2) m (k hw) c', m=self.num_heads)
        q_def = rearrange(queries_def,  'n p2 h w (m c)-> (n p2) m (h w) c',m=self.num_heads)
        attn_weight = (q_def * self.scale) @ k
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v
        out_def = rearrange(out, '(n j i) m (h w) c -> n (m c) (j h) (i w)', j=self.n_win, i=self.n_win, h=Hg//self.n_win, w=Wg//self.n_win).contiguous()
        out_def = out_def + lepe1
        out_def = self.unifyheads1(out_def)
        out_def = q_sampled + out_def
        out_def = out_def + self.mlp(self.norm2(out_def.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        out_def = self.norm(out_def.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        k = self.proj_k(out_def)
        v = self.proj_v(out_def)
        k_pix_sel = rearrange(k, 'n (m c) h w -> (n m) c (h w)', m=self.num_heads)
        v_pix_sel = rearrange(v, 'n (m c) h w -> (n m) c (h w)', m=self.num_heads)
        q_pix = rearrange(q, 'n (m c) h w -> (n m) c (h w)', m=self.num_heads)
        attn = torch.einsum('b c m, b c n -> b m n', q_pix, k_pix_sel).mul(self.scale)
        rpe_bias = self.rpe_table[None, ...].expand(N, -1, -1, -1)
        q_grid = self._get_q_grid(H, W, N, dtype, device)
        displacement = (q_grid.reshape(N * self.n_groups, H * W, 2).unsqueeze(2) - pos_k.reshape(N * self.n_groups, Hg*Wg, 2).unsqueeze(1)).mul(0.5)
        attn_bias = F.grid_sample(input=rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups), grid=displacement[..., (1, 0)], mode='bilinear', align_corners=True)
        attn_bias = attn_bias.reshape(N * self.num_heads, H * W, Hg*Wg)
        attn = F.softmax(attn + attn_bias, dim=2)
        out = torch.einsum('b m n, b c n -> b c m', attn, v_pix_sel).reshape(N,C,H,W).contiguous()
        out = self.proj_out(out).permute(0,2,3,1)
        if self.auto_pad and (pad_r > 0 or pad_b > 0): out = out[:, :H_in, :W_in, :].contiguous()
        return out

class DeBiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None, kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool', topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, mlp_ratio=4, param_size='small',mlp_dwconv=False, side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=True):
        super().__init__()
        self.attn2 = DeBiLevelRoutingAttentionblcok(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim, qk_scale=qk_scale, kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio, kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode, topk=topk, param_attention=param_attention, param_routing=param_routing, diff_routing=diff_routing, soft_routing=soft_routing, side_dwconv=side_dwconv, auto_pad=auto_pad,param_size=param_size)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.attn2(x)
        return x.permute(0, 3, 1, 2)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class DSAM(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.channel_attention = DeBiLevelRoutingAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))

class PSABlock_DSAM(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__()
        self.attn = DeBiLevelRoutingAttention(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut
    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA_DSAM(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock_DSAM(self.c) for _ in range(n)))
    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((a, self.m(b)), 1))