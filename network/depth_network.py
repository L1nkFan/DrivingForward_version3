from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import upsample, conv2d, pack_cam_feat, unpack_cam_feat
from .volumetric_fusionnet import VFNet


def _resolve_num_heads(channels, preferred_heads=4):
    """Pick a valid attention head count that divides channel size."""
    for heads in (preferred_heads, 8, 4, 2, 1):
        if heads > 0 and channels % heads == 0:
            return heads
    return 1


class CrossViewFusionBlock(nn.Module):
    """
    Lightweight cross-view attention over camera tokens.

    Input shape: [B, N_cam, C, H, W]
    Output shape: [B, N_cam, C, H, W]
    """

    def __init__(self, channels, num_cams=6, num_heads=4):
        super().__init__()
        heads = _resolve_num_heads(channels, preferred_heads=num_heads)
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.cam_embed = nn.Embedding(max(12, int(num_cams)), channels)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"CrossViewFusionBlock expects 5D input, got shape {tuple(x.shape)}")
        b, n_cam, c, _, _ = x.shape
        pooled = x.mean(dim=(-1, -2))
        cam_ids = torch.arange(n_cam, device=x.device)
        cam_bias = self.cam_embed(cam_ids).to(dtype=pooled.dtype).unsqueeze(0).expand(b, -1, -1)

        tokens = self.norm(pooled + cam_bias)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        fused = self.proj(torch.cat([tokens, attn_out], dim=-1))
        gate = self.gate(fused)
        fused = pooled + gate * fused

        return x + fused.unsqueeze(-1).unsqueeze(-1)


def _window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, c)
    return windows


def _window_reverse(windows, window_size, h, w):
    num_windows = (h // window_size) * (w // window_size)
    b = windows.shape[0] // num_windows
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(b, h, w, -1)
    return x


def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        self.attn_mask = None
        self.attn_mask_hw = None

    def _calculate_mask(self, h, w, device):
        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1
        mask_windows = _window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x, h, w):
        b, l, c = x.shape
        if l != h * w:
            raise ValueError("Input feature has wrong size.")

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        pad_b = (self.window_size - h % self.window_size) % self.window_size
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_r, 0, pad_b))
            x = x.permute(0, 2, 3, 1)
        h_pad, w_pad = h + pad_b, w + pad_r

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.attn_mask is None or self.attn_mask_hw != (h_pad, w_pad):
                self.attn_mask = self._calculate_mask(h_pad, w_pad, x.device)
                self.attn_mask_hw = (h_pad, w_pad)
        else:
            self.attn_mask = None

        x_windows = _window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)

        x = _window_reverse(attn_windows, self.window_size, h_pad, w_pad)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_b > 0 or pad_r > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, h, w):
        b, l, c = x.shape
        if l != h * w:
            raise ValueError("Input feature has wrong size.")

        x = x.view(b, h, w, c)
        pad_b = h % 2
        pad_r = w % 2
        if pad_b or pad_r:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_r, 0, pad_b))
            x = x.permute(0, 2, 3, 1)
            h = h + pad_b
            w = w + pad_r

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(b, -1, 4 * c)
        x = self.norm(x)
        x = self.reduction(x)
        return x, h // 2, w // 2


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        if isinstance(drop_path, list):
            dpr = drop_path
        else:
            dpr = [drop_path for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            self.blocks.append(block)

        self.norm = norm_layer(dim)
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x, h, w):
        for blk in self.blocks:
            x = blk(x, h, w)

        x_out = self.norm(x)
        h_out, w_out = h, w
        if self.downsample is not None:
            x, h, w = self.downsample(x_out, h, w)
        return x_out, h_out, w_out, x, h, w


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        b, c, h, w = x.shape
        pad_b = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_r = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, h, w


class SwinEncoder(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        norm_layer = nn.LayerNorm
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        cur = 0
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur:cur + depths[i_layer]],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)
            cur += depths[i_layer]

        # 1x1 adapters to match ResNet-18 channel dimensions
        stage_dims = [embed_dim * (2 ** i) for i in range(self.num_layers)]
        self.stage_adapters = nn.ModuleList([
            conv2d(stage_dims[0], 64, kernel_size=1, nonlin=None),
            conv2d(stage_dims[1], 128, kernel_size=1, nonlin=None),
            conv2d(stage_dims[2], 256, kernel_size=1, nonlin=None),
            conv2d(stage_dims[3], 512, kernel_size=1, nonlin=None),
        ])
        self.up_level0 = conv2d(64, 64, kernel_size=1, nonlin=None)

        self.num_ch_enc = [64, 64, 128, 256, 512]

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        x = self.pos_drop(x)

        stage_outputs = []
        for layer in self.layers:
            x_out, h_out, w_out, x, h, w = layer(x, h, w)
            stage_outputs.append((x_out, h_out, w_out))

        feats = []
        for idx, (x_out, h_out, w_out) in enumerate(stage_outputs):
            b, _, c = x_out.shape
            feat = x_out.view(b, h_out, w_out, c).permute(0, 3, 1, 2).contiguous()
            feat = self.stage_adapters[idx](feat)
            feats.append(feat)

        feat1, feat2, feat3, feat4 = feats
        feat0 = self.up_level0(F.interpolate(feat1, scale_factor=2, mode="bilinear", align_corners=True))
        return [feat0, feat1, feat2, feat3, feat4]

class DepthNetwork(nn.Module):
    """
    Depth fusion module
    """    
    def __init__(self, cfg):
        super(DepthNetwork, self).__init__()
        self.read_config(cfg)
        
        # feature encoder (Swin Transformer + adapters)
        self.swin_patch_size = getattr(self, "swin_patch_size", getattr(self, "vit_patch_size", 4))
        self.swin_embed_dim = getattr(self, "swin_embed_dim", 96)
        self.swin_depths = getattr(self, "swin_depths", [2, 2, 6, 2])
        self.swin_num_heads = getattr(self, "swin_num_heads", [3, 6, 12, 24])
        self.swin_window_size = getattr(self, "swin_window_size", 7)
        self.swin_mlp_ratio = getattr(self, "swin_mlp_ratio", 4.0)
        self.swin_drop_rate = getattr(self, "swin_drop_rate", 0.0)
        self.swin_attn_drop_rate = getattr(self, "swin_attn_drop_rate", 0.0)
        self.swin_drop_path_rate = getattr(self, "swin_drop_path_rate", 0.1)
        self.swin_patch_norm = getattr(self, "swin_patch_norm", True)

        depths = tuple(self.swin_depths) if isinstance(self.swin_depths, (list, tuple)) else (2, 2, 6, 2)
        num_heads = tuple(self.swin_num_heads) if isinstance(self.swin_num_heads, (list, tuple)) else (3, 6, 12, 24)

        self.encoder = SwinEncoder(
            img_size=(int(self.height), int(self.width)),
            patch_size=self.swin_patch_size,
            in_chans=3,
            embed_dim=self.swin_embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=self.swin_window_size,
            mlp_ratio=self.swin_mlp_ratio,
            drop_rate=self.swin_drop_rate,
            attn_drop_rate=self.swin_attn_drop_rate,
            drop_path_rate=self.swin_drop_path_rate,
            patch_norm=self.swin_patch_norm,
        )
        enc_feat_dim = sum(self.encoder.num_ch_enc[self.fusion_level:]) 
        self.conv1x1 = conv2d(enc_feat_dim, self.fusion_feat_in_dim, kernel_size=1, padding_mode = 'reflect') 

        # Cross-view fusion: share one backbone while explicitly exchanging information across cameras.
        self.enable_cross_view_fusion = getattr(self, "enable_cross_view_fusion", True)
        self.cross_view_num_heads = getattr(self, "cross_view_num_heads", 4)
        self.cross_view_fusion_agg = CrossViewFusionBlock(
            channels=self.fusion_feat_in_dim,
            num_cams=self.num_cams,
            num_heads=self.cross_view_num_heads,
        )
        img_feat_dims = self.encoder.num_ch_enc[:self.fusion_level] + [self.encoder.num_ch_enc[self.fusion_level]]
        self.cross_view_fusion_img = nn.ModuleList([
            CrossViewFusionBlock(channels=ch, num_cams=self.num_cams, num_heads=self.cross_view_num_heads)
            for ch in img_feat_dims
        ])

        # fusion net
        fusion_feat_out_dim = self.encoder.num_ch_enc[self.fusion_level] 
        self.fusion_net = VFNet(cfg, self.fusion_feat_in_dim, fusion_feat_out_dim, model ='depth')
        
        # depth decoder
        num_ch_enc = self.encoder.num_ch_enc[:(self.fusion_level+1)] 
        num_ch_dec = [16, 32, 64, 128, 256]
        self.decoder = DepthDecoder(self.fusion_level, num_ch_enc, num_ch_dec, self.scales, use_skips = self.use_skips)
    
    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def vit_parameters(self):
        return list(self.encoder.parameters())

    def _apply_cross_view_fusion(self, feat):
        """Apply cross-view fusion on [B, N_cam, C, H, W] tensor when enabled."""
        if not self.enable_cross_view_fusion:
            return feat
        return self.cross_view_fusion_agg(feat)

    def forward(self, inputs):
        '''
        dict_keys(['idx', 'sensor_name', 'filename', 'extrinsics', 'mask', 
        ('K', 0), ('inv_K', 0), ('color', 0, 0), ('color_aug', 0, 0), 
        ('K', 1), ('inv_K', 1), ('color', 0, 1), ('color_aug', 0, 1), 
        ('K', 2), ('inv_K', 2), ('color', 0, 2), ('color_aug', 0, 2), 
        ('K', 3), ('inv_K', 3), ('color', 0, 3), ('color_aug', 0, 3), 
        ('color', -1, 0), ('color_aug', -1, 0), ('color', 1, 0), ('color_aug', 1, 0), 'extrinsics_inv'])
        '''

        outputs = {}
        
        # dictionary initialize
        for cam in range(self.num_cams): # self.num_cames = 6
            outputs[('cam', cam)] = {} # outputs = {('cam', 0): {}, ..., ('cam', 5): {}}
        
        lev = self.fusion_level # 2
        
        # packed images for surrounding view
        sf_images = torch.stack([inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(self.num_cams)], 1) 
        if self.novel_view_mode == 'MF':
            sf_images_last = torch.stack([inputs[('color_aug', -1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
            sf_images_next = torch.stack([inputs[('color_aug', 1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        packed_input = pack_cam_feat(sf_images) 
        if self.novel_view_mode == 'MF':
            packed_input_last = pack_cam_feat(sf_images_last)
            packed_input_next = pack_cam_feat(sf_images_next)
        
        # feature encoder
        packed_feats = self.encoder(packed_input) 
        if self.novel_view_mode == 'MF':
            packed_feats_last = self.encoder(packed_input_last)
            packed_feats_next = self.encoder(packed_input_next)
        # aggregate feature H / 2^(lev+1) x W / 2^(lev+1)
        _, _, up_h, up_w = packed_feats[lev].size() 
        
        packed_feats_list = packed_feats[lev:lev+1] \
                        + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats[lev+1:]]        
        if self.novel_view_mode == 'MF':
            packed_feats_last_list = packed_feats_last[lev:lev+1] \
                            + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats_last[lev+1:]]
            packed_feats_next_list = packed_feats_next[lev:lev+1] \
                            + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats_next[lev+1:]]                

        packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1)) 
        if self.novel_view_mode == 'MF':
            packed_feats_agg_last = self.conv1x1(torch.cat(packed_feats_last_list, dim=1))
            packed_feats_agg_next = self.conv1x1(torch.cat(packed_feats_next_list, dim=1))

        feats_agg = unpack_cam_feat(packed_feats_agg, self.batch_size, self.num_cams) 
        if self.novel_view_mode == 'MF':
            feats_agg_last = unpack_cam_feat(packed_feats_agg_last, self.batch_size, self.num_cams)
            feats_agg_next = unpack_cam_feat(packed_feats_agg_next, self.batch_size, self.num_cams)

        # Explicitly fuse camera tokens before voxel projection.
        feats_agg = self._apply_cross_view_fusion(feats_agg)
        if self.novel_view_mode == 'MF':
            feats_agg_last = self._apply_cross_view_fusion(feats_agg_last)
            feats_agg_next = self._apply_cross_view_fusion(feats_agg_next)

        # fusion_net, backproject each feature into the 3D voxel space
        fusion_dict = self.fusion_net(inputs, feats_agg)
        if self.novel_view_mode == 'MF':
            fusion_dict_last = self.fusion_net(inputs, feats_agg_last)
            fusion_dict_next = self.fusion_net(inputs, feats_agg_next)

        feat_in = packed_feats[:lev] + [fusion_dict['proj_feat']]   
        img_feat = []
        for i in range(len(feat_in)):
            img_feat.append(unpack_cam_feat(feat_in[i], self.batch_size, self.num_cams))
            if self.enable_cross_view_fusion:
                img_feat[i] = self.cross_view_fusion_img[i](img_feat[i])
        
        if self.novel_view_mode == 'MF':
            feat_in_last = packed_feats_last[:lev] + [fusion_dict_last['proj_feat']] 
            img_feat_last = []
            for i in range(len(feat_in_last)):
                img_feat_last.append(unpack_cam_feat(feat_in_last[i], self.batch_size, self.num_cams))
                if self.enable_cross_view_fusion:
                    img_feat_last[i] = self.cross_view_fusion_img[i](img_feat_last[i])
            
            feat_in_next = packed_feats_next[:lev] + [fusion_dict_next['proj_feat']] 
            img_feat_next = []
            for i in range(len(feat_in_next)):
                img_feat_next.append(unpack_cam_feat(feat_in_next[i], self.batch_size, self.num_cams))
                if self.enable_cross_view_fusion:
                    img_feat_next[i] = self.cross_view_fusion_img[i](img_feat_next[i])

        packed_depth_outputs = self.decoder(feat_in)  
        if self.novel_view_mode == 'MF':      
            packed_depth_outputs_last = self.decoder(feat_in_last)
            packed_depth_outputs_next = self.decoder(feat_in_next)

        depth_outputs = unpack_cam_feat(packed_depth_outputs, self.batch_size, self.num_cams) 
        if self.novel_view_mode == 'MF':
            depth_outputs_last = unpack_cam_feat(packed_depth_outputs_last, self.batch_size, self.num_cams)
            depth_outputs_next = unpack_cam_feat(packed_depth_outputs_next, self.batch_size, self.num_cams)

        for cam in range(self.num_cams):
            for k in depth_outputs.keys():
                outputs[('cam', cam)][k] = depth_outputs[k][:, cam, ...]
            outputs[('cam', cam)][('img_feat', 0, 0)] = [feat[:, cam, ...] for feat in img_feat] 
            if self.novel_view_mode == 'MF':
                outputs[('cam', cam)][('img_feat', -1, 0)] = [feat[:, cam, ...] for feat in img_feat_last] 
                outputs[('cam', cam)][('img_feat', 1, 0)] = [feat[:, cam, ...] for feat in img_feat_next]
                outputs[('cam', cam)][('disp', -1, 0)] = depth_outputs_last[('disp', 0)][:, cam, ...] 
                outputs[('cam', cam)][('disp', 1, 0)] = depth_outputs_next[('disp', 0)][:, cam, ...] 

        return outputs
    
        
class DepthDecoder(nn.Module):
    def __init__(self, level_in, num_ch_enc, num_ch_dec, scales=range(2), use_skips=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = 1
        self.scales = scales
        self.use_skips = use_skips
        
        self.level_in = level_in
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec

        self.convs = OrderedDict()
        for i in range(self.level_in, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == self.level_in else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin = 'ELU')

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 1)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin = 'ELU')

        for s in self.scales:
            self.convs[('dispconv', s)] = conv2d(self.num_ch_dec[s], self.num_output_channels, 3, nonlin = None)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        
        # decode
        x = input_features[-1]
        for i in range(self.level_in, -1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            if i in self.scales:
                outputs[('disp', i)] = self.sigmoid(self.convs[('dispconv', i)](x))                
        return outputs
