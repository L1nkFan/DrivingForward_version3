"""Stage-2 multi-mode loss module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS


class Stage2LossMultiMode(nn.Module):

    def __init__(self,
                 lambda_warp=0.02,
                 lambda_consist=1e-5,
                 lambda_render=0.01,
                 enable_spatial_consistency=False,
                 enable_render_lpips=False,
                 rank=0):
        super(Stage2LossMultiMode, self).__init__()

        self.lambda_warp = lambda_warp
        self.lambda_consist = lambda_consist
        self.lambda_render = lambda_render
        self.rank = rank
        self.enable_spatial_consistency = enable_spatial_consistency
        self.enable_render_lpips = enable_render_lpips

        self.lpips = LPIPS(net="vgg").cuda(rank)

    def compute_l1_loss(self, pred, target, mask=None):
        l1_loss = torch.abs(pred - target)
        if mask is not None:
            l1_loss = l1_loss * mask
            return l1_loss.sum() / (mask.sum() + 1e-8)
        return l1_loss.mean()

    def compute_l2_loss(self, pred, target, mask=None):
        l2_loss = (pred - target) ** 2
        if mask is not None:
            l2_loss = l2_loss * mask
            return l2_loss.sum() / (mask.sum() + 1e-8)
        return l2_loss.mean()

    def compute_ssim_loss(self, pred, target, mask=None):
        original_shape = pred.shape
        if pred.dim() == 5:
            B, N, C, H, W = pred.shape
            pred = pred.view(B * N, C, H, W)
            target = target.view(B * N, C, H, W)
            if mask is not None:
                if mask.dim() == 5:
                    mask = mask.view(B * N, mask.shape[2], H, W)
                elif mask.dim() == 4:
                    mask = mask.view(B * N, mask.shape[1], H, W)

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

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred + sigma_target + C2) + 1e-8)

        ssim_loss = torch.clamp((1 - ssim_map) / 2, 0, 1)

        if mask is not None:
            ssim_loss = ssim_loss * mask
            return ssim_loss.sum() / (mask.sum() + 1e-8)
        return ssim_loss.mean()

    def compute_lpips_loss(self, pred, target):
        # Guard LPIPS against NaN/Inf from aggressive warping outputs.
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        lp = self.lpips(pred, target, normalize=True)
        lp = torch.nan_to_num(lp, nan=0.0, posinf=0.0, neginf=0.0)
        return lp.mean()

    def compute_warp_loss(self,
                          I_tgt,
                          I_src,
                          F_total,
                          mask,
                          use_lpips=True):
        from .rigid_flow import warp_image_with_flow, batch_warp_image_with_flow

        if I_tgt.dim() == 5:
            I_warped = batch_warp_image_with_flow(I_src, F_total)
        else:
            I_warped = warp_image_with_flow(I_src, F_total)

        if mask.dim() == 5:
            mask_expanded = mask.expand(-1, -1, 3, -1, -1)
        elif mask.dim() == 4 and mask.shape[1] == 1:
            mask_expanded = mask.expand(-1, 3, -1, -1)
        else:
            mask_expanded = mask

        # Guard against invalid values propagated by warping grids/flows.
        I_warped = torch.nan_to_num(I_warped, nan=0.0, posinf=1.0, neginf=0.0)
        I_tgt = torch.nan_to_num(I_tgt, nan=0.0, posinf=1.0, neginf=0.0)

        l1_loss = self.compute_l1_loss(I_warped, I_tgt, mask_expanded)

        ssim_loss = self.compute_ssim_loss(I_warped, I_tgt, mask_expanded)

        loss = l1_loss + 0.1 * ssim_loss

        if use_lpips:
            if I_tgt.dim() == 5:
                B, N, C, H, W = I_tgt.shape
                I_tgt_flat = I_tgt.view(B * N, C, H, W)
                I_warped_flat = I_warped.view(B * N, C, H, W)
                lpips_loss = self.compute_lpips_loss(I_warped_flat, I_tgt_flat)
            else:
                lpips_loss = self.compute_lpips_loss(I_warped, I_tgt)
            loss = loss + 0.05 * lpips_loss

        return loss

    def compute_flow_consistency_loss(self,
                                       I_src,
                                       I_tgt,
                                       F_forward,
                                       res_flow_net,
                                       mask,
                                       rigid_flow_fn=None):
        from .rigid_flow import warp_image_with_flow, batch_warp_image_with_flow

        if I_src.dim() == 5:
            F_rigid_backward = torch.zeros_like(F_forward)
            I_tgt_warped = batch_warp_image_with_flow(I_tgt, F_rigid_backward)
            F_residual_backward = res_flow_net(I_tgt_warped, I_src, F_rigid_backward)
            F_backward = F_rigid_backward + F_residual_backward
            F_forward_warped = batch_warp_image_with_flow(F_forward, F_backward)

            if mask.dim() == 5:
                mask_expanded = mask.expand(-1, -1, 2, -1, -1)
            else:
                mask_expanded = mask.unsqueeze(2).expand(-1, -1, 2, -1, -1)

            consistency_loss = self.compute_l1_loss(F_forward_warped + F_backward,
                                                     torch.zeros_like(F_forward),
                                                     mask_expanded)
        else:
            F_rigid_backward = torch.zeros_like(F_forward)
            I_tgt_warped = warp_image_with_flow(I_tgt, F_rigid_backward)
            F_residual_backward = res_flow_net(I_tgt_warped, I_src, F_rigid_backward)
            F_backward = F_rigid_backward + F_residual_backward
            F_forward_warped = warp_image_with_flow(F_forward, F_backward)

            if mask.dim() == 4 and mask.shape[1] == 1:
                mask_expanded = mask.expand(-1, 2, -1, -1)
            else:
                mask_expanded = mask

            consistency_loss = self.compute_l1_loss(F_forward_warped + F_backward,
                                                     torch.zeros_like(F_forward),
                                                     mask_expanded)

        return consistency_loss

    def compute_render_loss(self,
                            rendered_img,
                            gt_img,
                            use_lpips=True):
        l2_loss = self.compute_l2_loss(rendered_img, gt_img)
        loss = l2_loss

        if use_lpips:
            if rendered_img.dim() == 5:
                B, N, C, H, W = rendered_img.shape
                rendered_flat = rendered_img.view(B * N, C, H, W)
                gt_flat = gt_img.view(B * N, C, H, W)
                lpips_loss = self.compute_lpips_loss(rendered_flat, gt_flat)
            else:
                lpips_loss = self.compute_lpips_loss(rendered_img, gt_img)
            loss = loss + 0.05 * lpips_loss

        return loss

    def compute_temporal_loss(self,
                              I_t,
                              I_t_minus_1,
                              I_t_plus_1,
                              F_total_t_minus_1,
                              F_total_t_plus_1,
                              mask_t_minus_1,
                              mask_t_plus_1,
                              res_flow_net,
                              rigid_flow_fn=None):
        loss_warp_t_minus_1 = self.compute_warp_loss(
            I_t, I_t_minus_1, F_total_t_minus_1, mask_t_minus_1
        )
        loss_warp_t_plus_1 = self.compute_warp_loss(
            I_t, I_t_plus_1, F_total_t_plus_1, mask_t_plus_1
        )
        loss_warp = (loss_warp_t_minus_1 + loss_warp_t_plus_1) / 2.0

        if res_flow_net is not None:
            loss_consist_t_minus_1 = self.compute_flow_consistency_loss(
                I_t_minus_1, I_t, F_total_t_minus_1,
                res_flow_net, mask_t_minus_1, rigid_flow_fn
            )
            loss_consist_t_plus_1 = self.compute_flow_consistency_loss(
                I_t_plus_1, I_t, F_total_t_plus_1,
                res_flow_net, mask_t_plus_1, rigid_flow_fn
            )
            loss_consist = (loss_consist_t_minus_1 + loss_consist_t_plus_1) / 2.0
        else:
            loss_consist = torch.tensor(0.0, device=I_t.device)

        return {
            'loss_warp': loss_warp,
            'loss_consist': loss_consist,
            'loss_warp_t_minus_1': loss_warp_t_minus_1,
            'loss_warp_t_plus_1': loss_warp_t_plus_1,
        }

    def compute_spatio_loss(self,
                            I_current,
                            F_total_spatial,
                            mask_spatial,
                            src_cam_list,
                            tgt_cam_list,
                            res_flow_net,
                            rigid_flow_fn=None):
        if F_total_spatial is None or len(src_cam_list) == 0:
            return {
                'loss_warp_spatial': torch.tensor(0.0, device=I_current.device),
                'loss_consist_spatial': torch.tensor(0.0, device=I_current.device),
            }

        from .rigid_flow import batch_warp_image_with_flow

        batch_size, num_cams, C, H, W = I_current.shape
        num_pairs = F_total_spatial.shape[1]

        I_src_list = []
        I_tgt_list = []
        for src_cam, tgt_cam in zip(src_cam_list, tgt_cam_list):
            I_src_list.append(I_current[:, src_cam, ...])
            I_tgt_list.append(I_current[:, tgt_cam, ...])

        I_src = torch.stack(I_src_list, dim=1)  # [B, M, 3, H, W]
        I_tgt = torch.stack(I_tgt_list, dim=1)  # [B, M, 3, H, W]

        loss_warp_spatial = self.compute_warp_loss(
            I_tgt, I_src, F_total_spatial, mask_spatial, use_lpips=False
        )

        if res_flow_net is not None and self.enable_spatial_consistency:
            loss_consist_spatial = self.compute_flow_consistency_loss(
                I_src, I_tgt, F_total_spatial,
                res_flow_net, mask_spatial, rigid_flow_fn
            )
        else:
            loss_consist_spatial = torch.tensor(0.0, device=I_current.device)

        return {
            'loss_warp_spatial': loss_warp_spatial,
            'loss_consist_spatial': loss_consist_spatial,
        }

    def forward(self,
                I_t,
                I_t_minus_1,
                I_t_plus_1,
                stage2_outputs,
                rendered_I_t,
                res_flow_net=None,
                rigid_flow_fn=None):
        mode = stage2_outputs.get('mode', 'spatio_temporal')

        if mode == 'temporal':
            temporal_losses = self.compute_temporal_loss(
                I_t, I_t_minus_1, I_t_plus_1,
                stage2_outputs['F_total_t_minus_1'],
                stage2_outputs['F_total_t_plus_1'],
                stage2_outputs['mask_t_minus_1'],
                stage2_outputs['mask_t_plus_1'],
                res_flow_net, rigid_flow_fn
            )

            loss_render = self.compute_render_loss(rendered_I_t, I_t, use_lpips=self.enable_render_lpips)

            total_loss = (self.lambda_warp * temporal_losses['loss_warp'] +
                          self.lambda_consist * temporal_losses['loss_consist'] +
                          self.lambda_render * loss_render)

            loss_dict = {
                'total_loss': total_loss,
                'loss_warp': temporal_losses['loss_warp'],
                'loss_consist': temporal_losses['loss_consist'],
                'loss_render': loss_render,
                'loss_warp_t_minus_1': temporal_losses['loss_warp_t_minus_1'],
                'loss_warp_t_plus_1': temporal_losses['loss_warp_t_plus_1'],
                'mode': 'temporal'
            }

        elif mode == 'spatio':
            spatio_losses = self.compute_spatio_loss(
                I_t,
                stage2_outputs['F_total_spatial'],
                stage2_outputs['mask_spatial'],
                stage2_outputs['src_cam_list'],
                stage2_outputs['tgt_cam_list'],
                res_flow_net, rigid_flow_fn
            )

            total_loss = (self.lambda_warp * spatio_losses['loss_warp_spatial'] +
                          self.lambda_consist * spatio_losses['loss_consist_spatial'])

            loss_dict = {
                'total_loss': total_loss,
                'loss_warp_spatial': spatio_losses['loss_warp_spatial'],
                'loss_consist_spatial': spatio_losses['loss_consist_spatial'],
                'loss_render': torch.tensor(0.0, device=I_t.device),
                'mode': 'spatio'
            }

        else:
            temporal_losses = self.compute_temporal_loss(
                I_t, I_t_minus_1, I_t_plus_1,
                stage2_outputs['F_total_t_minus_1'],
                stage2_outputs['F_total_t_plus_1'],
                stage2_outputs['mask_t_minus_1'],
                stage2_outputs['mask_t_plus_1'],
                res_flow_net, rigid_flow_fn
            )

            spatio_losses = self.compute_spatio_loss(
                I_t,
                stage2_outputs.get('F_total_spatial'),
                stage2_outputs.get('mask_spatial'),
                stage2_outputs.get('src_cam_list', []),
                stage2_outputs.get('tgt_cam_list', []),
                res_flow_net, rigid_flow_fn
            )

            loss_render = self.compute_render_loss(rendered_I_t, I_t, use_lpips=self.enable_render_lpips)

            loss_warp = temporal_losses['loss_warp'] + spatio_losses['loss_warp_spatial']
            loss_consist = temporal_losses['loss_consist'] + spatio_losses['loss_consist_spatial']

            total_loss = (self.lambda_warp * loss_warp +
                          self.lambda_consist * loss_consist +
                          self.lambda_render * loss_render)

            loss_dict = {
                'total_loss': total_loss,
                'loss_warp': loss_warp,
                'loss_warp_temporal': temporal_losses['loss_warp'],
                'loss_warp_spatial': spatio_losses['loss_warp_spatial'],
                'loss_consist': loss_consist,
                'loss_consist_temporal': temporal_losses['loss_consist'],
                'loss_consist_spatial': spatio_losses['loss_consist_spatial'],
                'loss_render': loss_render,
                'loss_warp_t_minus_1': temporal_losses['loss_warp_t_minus_1'],
                'loss_warp_t_plus_1': temporal_losses['loss_warp_t_plus_1'],
                'mode': 'spatio_temporal'
            }

        return loss_dict


def build_stage2_loss_multi_mode(cfg=None, rank=0):
    if cfg is not None:
        lambda_warp = cfg.get('lambda_warp', 0.02)
        lambda_consist = cfg.get('lambda_consist', 1e-5)
        lambda_render = cfg.get('lambda_render', 0.01)
        enable_spatial_consistency = cfg.get('enable_spatial_consistency', False)
        enable_render_lpips = cfg.get('enable_render_lpips', False)
    else:
        lambda_warp = 0.02
        lambda_consist = 1e-5
        lambda_render = 0.01
        enable_spatial_consistency = False
        enable_render_lpips = False

    return Stage2LossMultiMode(
        lambda_warp=lambda_warp,
        lambda_consist=lambda_consist,
        lambda_render=lambda_render,
        enable_spatial_consistency=enable_spatial_consistency,
        enable_render_lpips=enable_render_lpips,
        rank=rank
    )


