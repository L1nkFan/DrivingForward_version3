import torch
import torch.nn as nn
import torch.nn.functional as F

def _build_norm(channels):
    """Use GroupNorm to avoid BatchNorm running-stat side effects in recurrent/shared calls."""
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = _build_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = _build_norm(channels)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x):
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + identity)


class ConvStem(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            _build_norm(out_ch),
            nn.SiLU(inplace=False),
            ResidualBlock(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class ImageEncoder(nn.Module):
    """Shared image encoder for warped and target images."""

    def __init__(self, in_channels=3, base_channels=48):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        self.s1 = ConvStem(in_channels, c1, stride=2)   # 1/2
        self.s2 = ConvStem(c1, c2, stride=2)            # 1/4
        self.s3 = ConvStem(c2, c3, stride=2)            # 1/8
        self.s4 = ConvStem(c3, c4, stride=2)            # 1/16
        self.out_channels = (c1, c2, c3, c4)

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        return f1, f2, f3, f4


class FlowEncoder(nn.Module):
    """Encoder for rigid flow prior."""

    def __init__(self, in_channels=2, base_channels=24):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        self.s1 = ConvStem(in_channels, c1, stride=2)
        self.s2 = ConvStem(c1, c2, stride=2)
        self.s3 = ConvStem(c2, c3, stride=2)
        self.s4 = ConvStem(c3, c4, stride=2)
        self.out_channels = (c1, c2, c3, c4)

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        return f1, f2, f3, f4


class CrossAttentionFusion(nn.Module):
    """Cross-attention at coarse scale: query=target, key/value=warped."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.SiLU(inplace=False),
            nn.Linear(channels, channels),
        )

    def forward(self, warped_feat, target_feat):
        b, c, h, w = warped_feat.shape
        q = target_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        kv = warped_feat.flatten(2).transpose(1, 2)

        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(qn, kvn, kvn, need_weights=False)

        fused = torch.cat([q + attn_out, torch.abs(q - kv)], dim=-1)
        fused = self.proj(fused)
        fused = fused.transpose(1, 2).reshape(b, c, h, w)
        return fused


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _build_norm(out_channels),
            nn.SiLU(inplace=False),
            ResidualBlock(out_channels),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SharedEncoder(nn.Module):
    """Backward-compatible wrapper; not used directly by new ResFlowNet."""

    def __init__(self, in_channels=8, base_channels=64):
        super().__init__()
        self.encoder = ImageEncoder(in_channels=in_channels, base_channels=max(16, base_channels // 2))

    def forward(self, x):
        return self.encoder(x)


class CameraDecoder(nn.Module):
    """Backward-compatible wrapper; not used directly by new ResFlowNet."""

    def __init__(self, base_channels=64):
        super().__init__()
        c = max(16, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(inplace=False),
            nn.Conv2d(c, 2, 3, padding=1),
        )

    def forward(self, features):
        if isinstance(features, (list, tuple)):
            x = features[-1]
        else:
            x = features
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.head(x)


class ResFlowNet(nn.Module):
    """
    Residual flow predictor (replacement):
    - Shared two-stream encoder (warped/target)
    - Coarse cross-attention fusion
    - U-shaped decoder with rigid-flow skips
    - Lightweight iterative refinement head
    """

    def __init__(self, num_cams=6, base_channels=64, refine_iters=2):
        super().__init__()
        self.num_cams = num_cams
        self.refine_iters = refine_iters

        # Use a slightly narrower encoder for memory/speed balance.
        img_base = max(24, int(base_channels * 0.75))
        flow_base = max(16, int(base_channels * 0.375))

        self.img_encoder = ImageEncoder(in_channels=3, base_channels=img_base)
        self.flow_encoder = FlowEncoder(in_channels=2, base_channels=flow_base)

        i1, i2, i3, i4 = self.img_encoder.out_channels
        f1, f2, f3, f4 = self.flow_encoder.out_channels

        self.attn_fuse = CrossAttentionFusion(i4, num_heads=4)

        # Build multi-scale skip features from two streams + rigid prior.
        self.skip3_proj = nn.Sequential(
            nn.Conv2d(i3 * 2 + f3, i3, 3, padding=1, bias=False),
            _build_norm(i3),
            nn.SiLU(inplace=False),
        )
        self.skip2_proj = nn.Sequential(
            nn.Conv2d(i2 * 2 + f2, i2, 3, padding=1, bias=False),
            _build_norm(i2),
            nn.SiLU(inplace=False),
        )
        self.skip1_proj = nn.Sequential(
            nn.Conv2d(i1 * 2 + f1, i1, 3, padding=1, bias=False),
            _build_norm(i1),
            nn.SiLU(inplace=False),
        )

        self.coarse_fuse = nn.Sequential(
            nn.Conv2d(i4 + f4, i4, 3, padding=1, bias=False),
            _build_norm(i4),
            nn.SiLU(inplace=False),
            ResidualBlock(i4),
        )

        self.dec3 = DecoderBlock(i4, i3, i3)
        self.dec2 = DecoderBlock(i3, i2, i2)
        self.dec1 = DecoderBlock(i2, i1, i1)

        self.flow_head = nn.Sequential(
            nn.Conv2d(i1, i1, 3, padding=1, bias=False),
            _build_norm(i1),
            nn.SiLU(inplace=False),
            nn.Conv2d(i1, 2, 3, padding=1),
        )

        self.refine_head = nn.Sequential(
            nn.Conv2d(3 + 3 + 2 + 2, 64, 3, padding=1, bias=False),
            _build_norm(64),
            nn.SiLU(inplace=False),
            ResidualBlock(64),
            nn.Conv2d(64, 2, 3, padding=1),
        )

    def _forward_single_camera(self, warped_img, tgt_img, rigid_flow):
        w1, w2, w3, w4 = self.img_encoder(warped_img)
        t1, t2, t3, t4 = self.img_encoder(tgt_img)
        r1, r2, r3, r4 = self.flow_encoder(rigid_flow)

        # Coarse fusion.
        attn_feat = self.attn_fuse(w4, t4)
        x = self.coarse_fuse(torch.cat([attn_feat, r4], dim=1))

        # Skip fusion.
        s3 = self.skip3_proj(torch.cat([w3, t3, r3], dim=1))
        s2 = self.skip2_proj(torch.cat([w2, t2, r2], dim=1))
        s1 = self.skip1_proj(torch.cat([w1, t1, r1], dim=1))

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        flow = self.flow_head(x)
        flow = F.interpolate(flow, size=rigid_flow.shape[-2:], mode="bilinear", align_corners=True)

        # Lightweight iterative refinement at full resolution.
        for _ in range(self.refine_iters):
            delta = self.refine_head(torch.cat([warped_img, tgt_img, rigid_flow, flow], dim=1))
            flow = flow + delta

        return flow

    def forward(self, warped_img, tgt_img, rigid_flow):
        # Multi-camera mode: [B, N, C, H, W]
        if warped_img.dim() == 5:
            _, num_cams = warped_img.shape[:2]
            outputs = []
            for cam_idx in range(num_cams):
                outputs.append(
                    self._forward_single_camera(
                        warped_img[:, cam_idx],
                        tgt_img[:, cam_idx],
                        rigid_flow[:, cam_idx],
                    )
                )
            return torch.stack(outputs, dim=1)

        # Single-camera mode: [B, C, H, W]
        return self._forward_single_camera(warped_img, tgt_img, rigid_flow)


class ResFlowNetParallel(nn.Module):
    """Compatibility wrapper using the same new backbone."""

    def __init__(self, num_cams=6, base_channels=64):
        super().__init__()
        self.impl = ResFlowNet(num_cams=num_cams, base_channels=base_channels)

    def forward(self, warped_img, tgt_img, rigid_flow):
        return self.impl(warped_img, tgt_img, rigid_flow)


def build_res_flow_net(cfg=None, num_cams=6, base_channels=64):
    if cfg is not None:
        num_cams = cfg.get("num_cams", num_cams)
        base_channels = cfg.get("res_flow_base_channels", base_channels)
    return ResFlowNet(num_cams=num_cams, base_channels=base_channels)


