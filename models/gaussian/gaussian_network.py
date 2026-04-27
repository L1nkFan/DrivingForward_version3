import torch
from torch import nn
from .extractor import UnetExtractor, ResidualBlock
from einops import rearrange


def _resolve_num_heads(channels, preferred_heads=4):
    for heads in (preferred_heads, 8, 4, 2, 1):
        if heads > 0 and channels % heads == 0:
            return heads
    return 1


class CrossViewFusionBlock(nn.Module):
    """Lightweight attention fusion over camera tokens."""

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

    def forward(self, feat):
        # feat: [B, N_cam, C, H, W]
        if feat.dim() != 5:
            raise ValueError(f"CrossViewFusionBlock expects 5D input, got {tuple(feat.shape)}")
        b, n_cam, _, _, _ = feat.shape
        pooled = feat.mean(dim=(-1, -2))
        cam_ids = torch.arange(n_cam, device=feat.device)
        cam_bias = self.cam_embed(cam_ids).to(dtype=pooled.dtype).unsqueeze(0).expand(b, -1, -1)
        tokens = self.norm(pooled + cam_bias)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        fused = self.proj(torch.cat([tokens, attn_out], dim=-1))
        gate = self.gate(fused)
        fused = pooled + gate * fused
        return feat + fused.unsqueeze(-1).unsqueeze(-1)


class GaussianNetwork(nn.Module):
    def __init__(
        self,
        rgb_dim=3,
        depth_dim=1,
        norm_fn='group',
        num_cams=6,
        cross_view_num_heads=4,
        enable_cross_view_fusion=True,
    ):
        super().__init__()
        self.rgb_dims = [64, 64, 128]
        self.depth_dims = [32, 48, 96]
        self.decoder_dims = [48, 64, 96]
        self.head_dim = 32

        self.sh_degree = 4
        self.d_sh = (self.sh_degree + 1) ** 2

        self.enable_cross_view_fusion = enable_cross_view_fusion
        self.num_cams = num_cams

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2] + self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1] + self.depth_dims[1] + self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0] + self.depth_dims[0] + self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0] + rgb_dim + 1, self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=False)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.head_dim, 4, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.sh_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.head_dim, 3 * self.d_sh, kernel_size=1),
        )

        self.rgb_view_fusion = nn.ModuleList([
            CrossViewFusionBlock(ch, num_cams=num_cams, num_heads=cross_view_num_heads)
            for ch in self.rgb_dims
        ])
        self.depth_view_fusion = nn.ModuleList([
            CrossViewFusionBlock(ch, num_cams=num_cams, num_heads=cross_view_num_heads)
            for ch in self.depth_dims
        ])
        self.head_view_fusion = CrossViewFusionBlock(self.head_dim, num_cams=num_cams, num_heads=cross_view_num_heads)

    @staticmethod
    def _flatten_views(x):
        b, n = x.shape[:2]
        return x.reshape(b * n, *x.shape[2:]), b, n

    @staticmethod
    def _reshape_views(x, b, n):
        return x.reshape(b, n, *x.shape[1:])

    def forward(self, img, depth, img_feat):
        # Supports both single-view and multi-view:
        # single-view: img [B,3,H,W], depth [B,1,H,W], img_feat list([B,C,h,w])
        # multi-view:  img [B,N,3,H,W], depth [B,N,1,H,W], img_feat list([B,N,C,h,w])
        is_multi_view = img.dim() == 5

        if is_multi_view:
            if depth.dim() != 5:
                raise ValueError("depth must be 5D when img is multi-view")
            if not isinstance(img_feat, (list, tuple)) or len(img_feat) != 3:
                raise ValueError("img_feat must be a 3-level list/tuple for GaussianNetwork")

            img, b, n = self._flatten_views(img)
            depth, _, _ = self._flatten_views(depth)
            img_feat1, _, _ = self._flatten_views(img_feat[0])
            img_feat2, _, _ = self._flatten_views(img_feat[1])
            img_feat3, _, _ = self._flatten_views(img_feat[2])
        else:
            b, n = None, None
            img_feat1, img_feat2, img_feat3 = img_feat

        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)

        if is_multi_view and self.enable_cross_view_fusion:
            img_feat1 = self.rgb_view_fusion[0](self._reshape_views(img_feat1, b, n)).reshape(-1, img_feat1.shape[1], img_feat1.shape[2], img_feat1.shape[3])
            img_feat2 = self.rgb_view_fusion[1](self._reshape_views(img_feat2, b, n)).reshape(-1, img_feat2.shape[1], img_feat2.shape[2], img_feat2.shape[3])
            img_feat3 = self.rgb_view_fusion[2](self._reshape_views(img_feat3, b, n)).reshape(-1, img_feat3.shape[1], img_feat3.shape[2], img_feat3.shape[3])

            depth_feat1 = self.depth_view_fusion[0](self._reshape_views(depth_feat1, b, n)).reshape(-1, depth_feat1.shape[1], depth_feat1.shape[2], depth_feat1.shape[3])
            depth_feat2 = self.depth_view_fusion[1](self._reshape_views(depth_feat2, b, n)).reshape(-1, depth_feat2.shape[1], depth_feat2.shape[2], depth_feat2.shape[3])
            depth_feat3 = self.depth_view_fusion[2](self._reshape_views(depth_feat3, b, n)).reshape(-1, depth_feat3.shape[1], depth_feat3.shape[2], depth_feat3.shape[3])

        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)

        up3 = self.decoder3(feat3)
        up3 = self.up(up3)
        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)
        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))

        up1 = self.up(up1)
        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)

        if is_multi_view and self.enable_cross_view_fusion:
            out = self.head_view_fusion(self._reshape_views(out, b, n)).reshape(-1, out.shape[1], out.shape[2], out.shape[3])

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.01)

        # opacity head
        opacity_out = self.opacity_head(out)

        # sh head
        sh_out = self.sh_head(out)

        sh_out = rearrange(sh_out, "n c h w -> n (h w) c")
        sh_out = rearrange(sh_out, "... (srf c) -> ... srf () c", srf=1)
        sh_out = rearrange(sh_out, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh_out = sh_out * self.sh_mask

        if is_multi_view:
            rot_out = self._reshape_views(rot_out, b, n)
            scale_out = self._reshape_views(scale_out, b, n)
            opacity_out = self._reshape_views(opacity_out, b, n)
            sh_out = sh_out.reshape(b, n, *sh_out.shape[1:])

        return rot_out, scale_out, opacity_out, sh_out
