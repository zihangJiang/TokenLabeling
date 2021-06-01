"""
Modified from https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/backbones/vit.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import (Conv2d, Linear, build_activation_layer, build_norm_layer,
                      constant_init, kaiming_init, normal_init)
from mmcv.runner import _load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import DropPath, trunc_normal_
from functools import partial
from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
class Mlp(nn.Module):
    """MLP layer for Encoder block.

    Args:
        in_features(int): Input dimension for the first fully
            connected layer.
        hidden_features(int): Output dimension for the first fully
            connected layer.
        out_features(int): Output dementsion for the second fully
            connected layer.
        act_cfg(dict): Config dict for activation layer.
            Default: dict(type='GELU').
        drop(float): Drop rate for the dropout layer. Dropout rate has
            to be between 0 and 1. Default: 0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention layer for Encoder block.

    Args:
        dim (int): Dimension for the input vector.
        num_heads (int): Number of parallel attention heads.
        qkv_bias (bool): Enable bias for qkv if True. Default: False.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Drop rate for attention output weights.
            Default: 0.
        proj_drop (float): Drop rate for output weights. Default: 0.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads,
                                  c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Implements encoder block with residual connection.

    Args:
        dim (int): The feature dimension.
        num_heads (int): Number of parallel attention heads.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Drop rate for mlp output weights. Default: 0.
        attn_drop (float): Drop rate for attention output weights.
            Default: 0.
        proj_drop (float): Drop rate for attn layer output weights.
            Default: 0.
        drop_path (float): Drop rate for paths of model.
            Default: 0.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN', requires_grad=True).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        skip_lam (float): residual connection factor. Default: 1.0
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 with_cp=False,
                 skip_lam=1.0):
        super(Block, self).__init__()
        self.with_cp = with_cp
        _, self.norm1 = build_norm_layer(norm_cfg, dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop,
                              proj_drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        _, self.norm2 = build_norm_layer(norm_cfg, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)
        self.skip_lam=skip_lam

    def forward(self, x):

        def _inner_forward(x):
            out = x + self.drop_path(self.attn(self.norm1(x)))/self.skip_lam
            out = out + self.drop_path(self.mlp(self.norm2(out)))/self.skip_lam
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size (int | tuple): Input image size.
            default: 224.
        patch_size (int): Width and height for a patch.
            default: 16.
        in_channels (int): Input channels for images. Default: 3.
        embed_dim (int): The embedding dimension. Default: 768.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        elif isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            raise TypeError('img_size must be type of int or tuple')
        h, w = self.img_size
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.proj = Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)

class PatchEmbed4_2(nn.Module):
    """ 
    Image to Patch Embedding with 4 layer convolution
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, stem_dim=64):
        super().__init__()
        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_channels, stem_dim, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(stem_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(stem_dim, stem_dim, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(stem_dim)
        self.conv3 = nn.Conv2d(stem_dim, stem_dim, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(stem_dim)

        self.proj = nn.Conv2d(stem_dim, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x

@BACKBONES.register_module()
class VisionTransformer(nn.Module):
    """Vision transformer backbone.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for
        Image Recognition at Scale` - https://arxiv.org/abs/2010.11929

    Args:
        img_size (tuple): input image size. Default: (224, 224).
        patch_size (int, tuple): patch size. Default: 16.
        in_channels (int): number of input channels. Default: 3.
        embed_dim (int): embedding dimension. Default: 768.
        depth (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        qk_scale (float): override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): dropout rate. Default: 0.
        attn_drop_rate (float): attention dropout rate. Default: 0.
        drop_path_rate (float): Rate of DropPath. Default: 0.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN', eps=1e-6, requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        final_norm (bool):  Whether to add a additional layer to normalize
            final feature map. Default: False.
        out_reshape (str): Select the output format of feature information.
            Default: NCHW.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        with_cls_token (bool): If concatenating class token into image tokens
            as transformer input. Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        p_emb (str): Which Patch Embedding to use.
            Default: None, using naive Patch Embedding.
        stem_dim (int): hidden dim in Patch Embedding module.
            Default: 64.
        skip_lam (float): residual connection factor.
            Default: 1.0.
        use_side_layer (bool): whether use the side layer for UperNet and FCN.
            Default: False (use the neck instead)
        fcn (bool): switch between FCN and UperNet. 
            Default: False (use UperNet).
    """

    def __init__(self,
                 img_size=(224, 224),
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=11,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6, requires_grad=True),
                 act_cfg=dict(type='GELU'),
                 norm_eval=False,
                 final_norm=False,
                 out_shape='NCHW',
                 with_cls_token=True,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 out_channels=768,
                 p_emb=None,
                 stem_dim=64,
                 skip_lam=1.0,
                 use_side_layer=False,
                 fcn=False):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.features = self.embed_dim = embed_dim
        if p_emb=='4_2':
            patch_embed_fn = partial(PatchEmbed4_2,stem_dim=stem_dim)
        else:
            patch_embed_fn = PatchEmbed
        self.patch_embed = patch_embed_fn(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim)

        self.with_cls_token = with_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_patches = self.patch_embed.num_patches
        if isinstance(out_indices, int):
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dpr[i],
                attn_drop=attn_drop_rate,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                skip_lam=skip_lam) for i in range(depth)
        ])
        self.use_side_layer = use_side_layer
        if use_side_layer:
            if not fcn:
                self.side_layer1 = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=4, padding=0, bias=False),
                        nn.SyncBatchNorm(embed_dim),
                        nn.ReLU(True),
                        nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=False),
                        nn.SyncBatchNorm(embed_dim),
                        nn.ReLU(True),
                )
                self.side_layer2 = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, 2, stride=2, padding=0, bias=False),
                        nn.SyncBatchNorm(embed_dim),
                        nn.ReLU(True),
                        nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=False),
                        nn.SyncBatchNorm(embed_dim),
                        nn.ReLU(True),
                )
                self.side_layer3 = nn.Sequential(
                        nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=False),
                        nn.SyncBatchNorm(embed_dim),
                        nn.ReLU(True),
                )
                self.side_layer4 = nn.Sequential(
                        nn.Conv2d(embed_dim, out_channels, 1, 1, 0, bias=False),
                        nn.SyncBatchNorm(out_channels),
                        nn.ReLU(True),
                )
            else:
                self.side_layer1 = nn.Identity()
                self.side_layer2 = nn.Identity()
                self.side_layer3 = nn.Sequential(
                        nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=False),
                        nn.SyncBatchNorm(embed_dim),
                        nn.ReLU(True),
                )
                self.side_layer4 = nn.Sequential(
                        nn.ConvTranspose2d(embed_dim, embed_dim, 2, stride=2, padding=0, bias=False),
                        nn.SyncBatchNorm(embed_dim),
                        nn.ReLU(True),
                        nn.Conv2d(embed_dim, out_channels, 1, 1, 0, bias=False),
                        nn.SyncBatchNorm(out_channels),
                        nn.ReLU(True),
                )
        assert out_shape in ['NLC',
                             'NCHW'], 'output shape must be "NLC" or "NCHW".'

        self.out_shape = out_shape

        self.interpolate_mode = interpolate_mode
        self.final_norm = final_norm
        if final_norm:
            _, self.norm = build_norm_layer(norm_cfg, embed_dim)

        self.norm_eval = norm_eval
        self.with_cp = with_cp

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(pretrained, logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    logger.info(msg=f'Resize the pos_embed shape from \
{state_dict["pos_embed"].shape} to {self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'], (h, w), (pos_size, pos_size),
                        self.patch_size, self.interpolate_mode)

            self.load_state_dict(state_dict, False)

        elif pretrained is None:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'mlp' in n:
                            normal_init(m.bias, std=1e-6)
                        else:
                            constant_init(m.bias, 0)
                elif isinstance(m, Conv2d):
                    kaiming_init(m.weight, mode='fan_in')
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
        else:
            raise TypeError('pretrained must be a str or None')

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, patch_size, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): pos_embed weights.
            input_shpae (tuple): Tuple for (input_h, intput_w).
            pos_shape (tuple): Tuple for (pos_h, pos_w).
            patch_size (int): Patch size.
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        input_h, input_w = input_shpae
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(
            pos_embed_weight,
            size=[input_h // patch_size, input_w // patch_size],
            align_corners=False,
            mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def _pos_embeding(self, x, h, w):
        """Positiong embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            x (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
            h (int): training feature map height
            w (int): training feature map width
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        B,_,C = x.size()
        ct = x[:,0].unsqueeze(2)
        ts = x[:,1:].transpose(1, 2).reshape(B, C, int(self.num_patches ** 0.5), int(self.num_patches ** 0.5))
        ts = F.interpolate(ts, (h, w), mode='bicubic', align_corners=False)
        ts = ts.flatten(2)
        x = torch.cat([ct, ts], dim=2).transpose(1, 2)
        return x
    def forward(self, inputs):
        B = inputs.shape[0]

        x = self.patch_embed(inputs)
        B, C, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        #x = self._pos_embeding(inputs, x, self.pos_embed)
        x = x + self._pos_embeding(self.pos_embed, H, W)

        if not self.with_cls_token:
            # Remove class token for transformer input
            x = x[:, 1:]

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1:
                if self.final_norm:
                    x = self.norm(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                if self.out_shape == 'NCHW':
                    B, _, C = out.shape
                    out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
                outs.append(out)
        if self.use_side_layer:
            outs[0] = self.side_layer1(outs[0])
            outs[1] = self.side_layer2(outs[1])
            outs[2] = self.side_layer3(outs[2])
            outs[3] = self.side_layer4(outs[3])
        return tuple(outs)

    def train(self, mode=True):
        super(VisionTransformer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()

@BACKBONES.register_module()
class ViT(VisionTransformer):
    def __init__(self, **kwargs):
        super(ViT, self).__init__(**kwargs)