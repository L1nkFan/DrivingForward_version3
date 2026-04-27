import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS


class Stage2Loss(nn.Module):
    """Stage2 loss with robust photometric terms and dynamic weighting."""

    def __init__(self, lambda_warp=0.02, lambda_consist=1e-5, lambda_render=0.01, rank=0):
        super().__init__()
        self.lambda_warp = lambda_warp
        self.lambda_consist = lambda_consist
        self.lambda_render = lambda_render
        self.base_lambda_warp = lambda_warp
        self.base_lambda_consist = lambda_consist
        self.base_lambda_render = lambda_render
        self.rank = rank
        self.progress = 0.0
        self.lpips = LPIPS(net="vgg").cuda(rank)

    def set_training_progress(self, epoch, num_epochs, step=None, steps_per_epoch=None):
        """Update dynamic loss weights using epoch progress."""
        denom = max(1, num_epochs - 1)
        self.progress = float(epoch) / float(denom)
        self.lambda_warp = self.base_lambda_warp * (1.0 - 0.30 * self.progress)
        self.lambda_consist = self.base_lambda_consist * (1.0 + 1.00 * self.progress)
        self.lambda_render = self.base_lambda_render * (1.0 + 0.50 * self.progress)

    def compute_l1_loss(self, pred, target, mask=None):
        """Robust Charbonnier loss."""
        eps = 1e-3
        loss = torch.sqrt((pred - target) ** 2 + eps ** 2)
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_l2_loss(self, pred, target, mask=None):
        loss = (pred - target) ** 2
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_ssim_loss(self, pred, target, mask=None):
        if pred.dim() == 5:
            b, n, c, h, w = pred.shape
            pred = pred.view(b * n, c, h, w)
            target = target.view(b * n, c, h, w)
            if mask is not None:
                mask = mask.view(b * n, c, h, w)

        ref_pad = nn.ReflectionPad2d(1)
        pred = ref_pad(pred)
        target = ref_pad(target)

        mu_pred = F.avg_pool2d(pred, kernel_size=3, stride=1)
        mu_target = F.avg_pool2d(target, kernel_size=3, stride=1)

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred = F.avg_pool2d(pred.pow(2), kernel_size=3, stride=1) - mu_pred_sq
        sigma_target = F.avg_pool2d(target.pow(2), kernel_size=3, stride=1) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, kernel_size=3, stride=1) - mu_pred_target

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
            (mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2) + 1e-8
        )

        ssim_loss = torch.clamp((1 - ssim_map) / 2, 0, 1)
        if mask is not None:
            ssim_loss = ssim_loss * mask
            return ssim_loss.sum() / (mask.sum() + 1e-8)
        return ssim_loss.mean()

    def compute_ms_ssim_loss(self, pred, target, mask=None, levels=3):
        loss = 0.0
        cur_pred, cur_target, cur_mask = pred, target, mask
        for _ in range(levels):
            loss = loss + self.compute_ssim_loss(cur_pred, cur_target, cur_mask)
            if cur_pred.dim() == 5:
                cur_pred = F.avg_pool3d(cur_pred, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                cur_target = F.avg_pool3d(cur_target, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                if cur_mask is not None:
                    cur_mask = F.avg_pool3d(cur_mask.float(), kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                cur_pred = F.avg_pool2d(cur_pred, kernel_size=2, stride=2)
                cur_target = F.avg_pool2d(cur_target, kernel_size=2, stride=2)
                if cur_mask is not None:
                    cur_mask = F.avg_pool2d(cur_mask.float(), kernel_size=2, stride=2)
        return loss / float(levels)

    def compute_lpips_loss(self, pred, target):
        return self.lpips(pred, target, normalize=True).mean()

    def predict_backward_flow(self, I_src, I_tgt, res_flow_net):
        from .rigid_flow import warp_image_with_flow, batch_warp_image_with_flow

        if I_src.dim() == 5:
            f_rigid_backward = torch.zeros_like(I_src[..., :2, :, :])
            I_tgt_warped = batch_warp_image_with_flow(I_tgt, f_rigid_backward)
        else:
            f_rigid_backward = torch.zeros_like(I_src[:, :2, :, :])
            I_tgt_warped = warp_image_with_flow(I_tgt, f_rigid_backward)

        f_residual_backward = res_flow_net(I_tgt_warped, I_src, f_rigid_backward)
        return f_rigid_backward + f_residual_backward

    def compute_fb_consistency_mask(self, f_forward, f_backward, base_mask=None, tau=1.0):
        from .rigid_flow import warp_image_with_flow, batch_warp_image_with_flow

        if f_forward.dim() == 5:
            f_bwd_warp = batch_warp_image_with_flow(f_backward, f_forward)
            residual = torch.norm(f_forward + f_bwd_warp, p=1, dim=2, keepdim=True)
        else:
            f_bwd_warp = warp_image_with_flow(f_backward, f_forward)
            residual = torch.norm(f_forward + f_bwd_warp, p=1, dim=1, keepdim=True)

        visible = (residual < tau).float()
        if base_mask is not None:
            visible = visible * base_mask
        return visible

    def compute_warp_loss(self, I_tgt, I_src, f_total, mask, occlusion_mask=None, use_lpips=True):
        from .rigid_flow import warp_image_with_flow, batch_warp_image_with_flow

        if I_tgt.dim() == 5:
            I_warped = batch_warp_image_with_flow(I_src, f_total)
            mask_expanded = mask.expand(-1, -1, 3, -1, -1)
        else:
            I_warped = warp_image_with_flow(I_src, f_total)
            mask_expanded = mask.expand(-1, 3, -1, -1)

        if occlusion_mask is not None:
            mask_expanded = mask_expanded * occlusion_mask.expand_as(mask_expanded)

        l1_loss = self.compute_l1_loss(I_warped, I_tgt, mask_expanded)
        ms_ssim_loss = self.compute_ms_ssim_loss(I_warped, I_tgt, mask_expanded, levels=3)
        loss = l1_loss + 0.1 * ms_ssim_loss

        if use_lpips:
            if I_tgt.dim() == 5:
                b, n, c, h, w = I_tgt.shape
                lpips_loss = self.compute_lpips_loss(I_warped.view(b * n, c, h, w), I_tgt.view(b * n, c, h, w))
            else:
                lpips_loss = self.compute_lpips_loss(I_warped, I_tgt)
            loss = loss + 0.05 * lpips_loss

        return loss

    def compute_flow_consistency_loss(
        self,
        I_src,
        I_tgt,
        f_forward,
        res_flow_net,
        mask,
        rigid_flow_fn=None,
        occlusion_mask=None,
        f_backward=None,
    ):
        from .rigid_flow import warp_image_with_flow, batch_warp_image_with_flow

        if f_backward is None:
            f_backward = self.predict_backward_flow(I_src, I_tgt, res_flow_net)

        if I_src.dim() == 5:
            f_forward_warped = batch_warp_image_with_flow(f_forward, f_backward)
            fb_mask = self.compute_fb_consistency_mask(f_forward, f_backward, base_mask=mask)
            if occlusion_mask is not None:
                fb_mask = fb_mask * occlusion_mask
            mask_expanded = fb_mask.expand(-1, -1, 2, -1, -1)
        else:
            f_forward_warped = warp_image_with_flow(f_forward, f_backward)
            fb_mask = self.compute_fb_consistency_mask(f_forward, f_backward, base_mask=mask)
            if occlusion_mask is not None:
                fb_mask = fb_mask * occlusion_mask
            mask_expanded = fb_mask.expand(-1, 2, -1, -1)

        return self.compute_l1_loss(f_forward_warped + f_backward, torch.zeros_like(f_forward), mask_expanded)

    def compute_render_loss(self, rendered_img, gt_img, use_lpips=True):
        l2_loss = self.compute_l2_loss(rendered_img, gt_img)
        ms_ssim_loss = self.compute_ms_ssim_loss(rendered_img, gt_img, mask=None, levels=3)
        loss = l2_loss + 0.1 * ms_ssim_loss

        if use_lpips:
            if rendered_img.dim() == 5:
                b, n, c, h, w = rendered_img.shape
                lpips_loss = self.compute_lpips_loss(rendered_img.view(b * n, c, h, w), gt_img.view(b * n, c, h, w))
            else:
                lpips_loss = self.compute_lpips_loss(rendered_img, gt_img)
            loss = loss + 0.05 * lpips_loss

        return loss

    def forward(
        self,
        I_t,
        I_t_minus_1,
        I_t_plus_1,
        F_total_t_minus_1_to_t,
        F_total_t_plus_1_to_t,
        mask_t_minus_1,
        mask_t_plus_1,
        rendered_I_t,
        res_flow_net=None,
        rigid_flow_fn=None,
    ):
        # Estimate visibility masks with forward-backward consistency.
        occ_mask_t_minus_1 = None
        occ_mask_t_plus_1 = None
        f_bwd_for_minus = None
        f_bwd_for_plus = None
        if res_flow_net is not None:
            with torch.no_grad():
                f_bwd_for_minus = self.predict_backward_flow(I_t_minus_1, I_t, res_flow_net)
                f_bwd_for_plus = self.predict_backward_flow(I_t_plus_1, I_t, res_flow_net)
                occ_mask_t_minus_1 = self.compute_fb_consistency_mask(
                    F_total_t_minus_1_to_t, f_bwd_for_minus, base_mask=mask_t_minus_1
                )
                occ_mask_t_plus_1 = self.compute_fb_consistency_mask(
                    F_total_t_plus_1_to_t, f_bwd_for_plus, base_mask=mask_t_plus_1
                )

        loss_warp_t_minus_1 = self.compute_warp_loss(
            I_t, I_t_minus_1, F_total_t_minus_1_to_t, mask_t_minus_1, occlusion_mask=occ_mask_t_minus_1
        )
        loss_warp_t_plus_1 = self.compute_warp_loss(
            I_t, I_t_plus_1, F_total_t_plus_1_to_t, mask_t_plus_1, occlusion_mask=occ_mask_t_plus_1
        )
        loss_warp = (loss_warp_t_minus_1 + loss_warp_t_plus_1) / 2.0

        if res_flow_net is not None:
            loss_consist_t_minus_1 = self.compute_flow_consistency_loss(
                I_t_minus_1,
                I_t,
                F_total_t_minus_1_to_t,
                res_flow_net,
                mask_t_minus_1,
                rigid_flow_fn,
                occlusion_mask=occ_mask_t_minus_1,
                f_backward=f_bwd_for_minus,
            )
            loss_consist_t_plus_1 = self.compute_flow_consistency_loss(
                I_t_plus_1,
                I_t,
                F_total_t_plus_1_to_t,
                res_flow_net,
                mask_t_plus_1,
                rigid_flow_fn,
                occlusion_mask=occ_mask_t_plus_1,
                f_backward=f_bwd_for_plus,
            )
            loss_consist = (loss_consist_t_minus_1 + loss_consist_t_plus_1) / 2.0
        else:
            loss_consist = torch.tensor(0.0, device=I_t.device)

        loss_render = self.compute_render_loss(rendered_I_t, I_t)

        total_loss = self.lambda_warp * loss_warp + self.lambda_consist * loss_consist + self.lambda_render * loss_render

        return {
            'total_loss': total_loss,
            'loss_warp': loss_warp,
            'loss_consist': loss_consist,
            'loss_render': loss_render,
            'loss_warp_t_minus_1': loss_warp_t_minus_1,
            'loss_warp_t_plus_1': loss_warp_t_plus_1,
            'lambda_warp': torch.tensor(self.lambda_warp, device=I_t.device),
            'lambda_consist': torch.tensor(self.lambda_consist, device=I_t.device),
            'lambda_render': torch.tensor(self.lambda_render, device=I_t.device),
        }


def build_stage2_loss(cfg=None, rank=0):
    if cfg is not None:
        lambda_warp = cfg.get('lambda_warp', 0.02)
        lambda_consist = cfg.get('lambda_consist', 1e-5)
        lambda_render = cfg.get('lambda_render', 0.01)
    else:
        lambda_warp = 0.02
        lambda_consist = 1e-5
        lambda_render = 0.01

    return Stage2Loss(
        lambda_warp=lambda_warp,
        lambda_consist=lambda_consist,
        lambda_render=lambda_render,
        rank=rank,
    )
