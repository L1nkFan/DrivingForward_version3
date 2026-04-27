#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export Stage-2 rendered images for visual inspection.

Outputs:
- images/: gt / rendered(final)
- panels/: gt | rendered | diff-heatmap
- panels_gt_final/: gt | rendered(final)
- branch_images/: rendered_base / rendered_dynamic
- panels_base_dynamic/: gt | rendered_base | rendered_dynamic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader


def _setup_paths():
    root = Path(__file__).resolve().parents[1]
    extra_libs = [
        root,
        root / "external" / "packnet_sfm",
        root / "external" / "dgp",
        root / "external" / "monodepth2",
    ]
    for p in extra_libs:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    return root


def _parse_args():
    parser = argparse.ArgumentParser(description="Export Stage-2 rendered images")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/nuscenes/phase2_training_multi_mode.yaml",
        help="Path to stage2 config yaml",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to stage2 res_flow_net checkpoint (.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_stage2/vis_export",
        help="Directory to save exported images",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "eval"],
        help="Dataset split to export",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=200,
        help="Max dataloader batches to export, -1 for full split",
    )
    parser.add_argument(
        "--diff_gain",
        type=float,
        default=8.0,
        help="Amplification factor for diff heatmap visibility",
    )
    parser.add_argument(
        "--diff_gamma",
        type=float,
        default=0.7,
        help="Gamma for diff heatmap; <1 brightens subtle differences",
    )
    parser.add_argument(
        "--save_raw_npz",
        action="store_true",
        help="Save raw arrays for later quantitative analysis",
    )
    return parser.parse_args()


def _load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_uint8_hwc01(img_chw):
    arr = np.clip(img_chw, 0.0, 1.0).transpose(1, 2, 0)
    return (arr * 255.0).astype(np.uint8)


def _save_png_chw(img_chw, save_path):
    Image.fromarray(_to_uint8_hwc01(img_chw)).save(save_path)


def _make_diff_heatmap(gt_chw, pred_chw, diff_gain=8.0, diff_gamma=0.7):
    gt = np.clip(gt_chw, 0.0, 1.0).transpose(1, 2, 0).astype(np.float32)
    pred = np.clip(pred_chw, 0.0, 1.0).transpose(1, 2, 0).astype(np.float32)

    err = np.mean(np.abs(pred - gt), axis=2)
    err = np.clip(err * diff_gain, 0.0, 1.0)
    err = np.power(err, diff_gamma)

    heat = np.zeros((err.shape[0], err.shape[1], 3), dtype=np.float32)
    heat[..., 0] = np.clip(2.0 * err, 0.0, 1.0)
    heat[..., 1] = np.clip(2.0 * err - 0.35, 0.0, 1.0)
    heat[..., 2] = np.clip(1.0 - 2.0 * err, 0.0, 1.0)

    heat_u8 = (heat * 255.0).astype(np.uint8)
    mae = float(np.mean(np.abs(pred - gt)))
    return heat_u8, mae


def _save_panel(gt_chw, pred_chw, save_path, diff_gain=8.0, diff_gamma=0.7):
    gt = _to_uint8_hwc01(gt_chw)
    pred = _to_uint8_hwc01(pred_chw)
    diff_heat, mae = _make_diff_heatmap(gt_chw, pred_chw, diff_gain=diff_gain, diff_gamma=diff_gamma)
    panel = np.concatenate([gt, pred, diff_heat], axis=1)
    Image.fromarray(panel).save(save_path)
    return mae


def _save_gt_final_panel(gt_chw, final_chw, save_path):
    gt = _to_uint8_hwc01(gt_chw)
    final = _to_uint8_hwc01(final_chw)
    panel = np.concatenate([gt, final], axis=1)
    Image.fromarray(panel).save(save_path)


def _save_branch_panel(gt_chw, base_chw, dynamic_chw, save_path):
    gt = _to_uint8_hwc01(gt_chw)
    base = _to_uint8_hwc01(base_chw)
    dynamic = _to_uint8_hwc01(dynamic_chw)
    panel = np.concatenate([gt, base, dynamic], axis=1)
    Image.fromarray(panel).save(save_path)


def _compute_temporal_branches(inputs, outputs):
    """
    Recompute render branches for analysis only:
    - rendered_base: old mask-average branch
    - rendered_dynamic: residual-flow-aware branch
    """
    from stage2_modules.rigid_flow import batch_warp_image_with_flow

    f_m1 = outputs.get("F_total_t_minus_1")
    f_p1 = outputs.get("F_total_t_plus_1")
    if f_m1 is None or f_p1 is None:
        return None, None

    i_m1 = inputs[("color", -1, 0)]
    i_p1 = inputs[("color", 1, 0)]

    warped_m1 = batch_warp_image_with_flow(i_m1, f_m1)
    warped_p1 = batch_warp_image_with_flow(i_p1, f_p1)

    mask_m1 = outputs.get("mask_t_minus_1", torch.ones_like(f_m1[:, :, 0:1, :, :]))
    mask_p1 = outputs.get("mask_t_plus_1", torch.ones_like(f_p1[:, :, 0:1, :, :]))

    # Base branch: original behavior.
    w_m1 = mask_m1.expand(-1, -1, 3, -1, -1)
    w_p1 = mask_p1.expand(-1, -1, 3, -1, -1)
    w_sum = w_m1 + w_p1 + 1e-6
    rendered_base = (warped_m1 * w_m1 + warped_p1 * w_p1) / w_sum

    # Dynamic branch: residual-flow-aware confidence blend.
    res_m1 = outputs.get("F_residual_t_minus_1", torch.zeros_like(f_m1))
    res_p1 = outputs.get("F_residual_t_plus_1", torch.zeros_like(f_p1))
    res_mag_m1 = torch.linalg.norm(res_m1, dim=2, keepdim=True)
    res_mag_p1 = torch.linalg.norm(res_p1, dim=2, keepdim=True)

    flow_beta = 0.12
    conf_m1 = mask_m1 * torch.exp(-flow_beta * res_mag_m1)
    conf_p1 = mask_p1 * torch.exp(-flow_beta * res_mag_p1)

    conf_w_m1 = conf_m1.expand(-1, -1, 3, -1, -1)
    conf_w_p1 = conf_p1.expand(-1, -1, 3, -1, -1)
    conf_sum = conf_w_m1 + conf_w_p1 + 1e-6
    rendered_dynamic = (warped_m1 * conf_w_m1 + warped_p1 * conf_w_p1) / conf_sum

    return rendered_base, rendered_dynamic


def main():
    _setup_paths()
    args = _parse_args()

    from dataset import construct_dataset
    from stage2_trainer.model_factory import build_stage2_model

    cfg = _load_cfg(args.config_file)

    eval_augmentation = {
        "image_shape": (int(cfg["training"]["height"]), int(cfg["training"]["width"])),
        "jittering": (0.0, 0.0, 0.0, 0.0),
        "crop_train_borders": (),
        "crop_eval_borders": (),
    }

    dataset = construct_dataset(cfg, args.split, **eval_augmentation)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank = 0

    model = build_stage2_model(cfg, rank=rank)
    model.load_res_flow_net(args.checkpoint_path)
    model.set_val()
    model.to(device)

    out_root = Path(args.output_dir)
    img_dir = out_root / "images"
    panel_dir = out_root / "panels"
    gt_final_panel_dir = out_root / "panels_gt_final"
    branch_img_dir = out_root / "branch_images"
    branch_panel_dir = out_root / "panels_base_dynamic"
    raw_dir = out_root / "raw"

    img_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    gt_final_panel_dir.mkdir(parents=True, exist_ok=True)
    branch_img_dir.mkdir(parents=True, exist_ok=True)
    branch_panel_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw_npz:
        raw_dir.mkdir(parents=True, exist_ok=True)

    no_device_keys = {"idx", "dataset_idx", "sensor_name", "filename", "token"}

    exported = 0
    panel_maes = []

    print(f"[INFO] dataset size: {len(dataset)}")
    print(f"[INFO] checkpoint: {args.checkpoint_path}")
    print(f"[INFO] output dir: {out_root}")
    print(f"[INFO] diff settings: gain={args.diff_gain}, gamma={args.diff_gamma}")

    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, value in inputs.items():
                if key in no_device_keys:
                    continue
                if "context" in key or "ego_pose" in key:
                    inputs[key] = [value[k].float().to(device) for k in range(len(value))]
                else:
                    inputs[key] = value.float().to(device)

            outputs, _ = model(inputs)
            rendered = outputs["rendered_I_t"]
            gt = inputs[("color", 0, 0)]

            rendered_base, rendered_dynamic = _compute_temporal_branches(inputs, outputs)
            if rendered_base is None or rendered_dynamic is None:
                rendered_base = rendered
                rendered_dynamic = rendered

            rendered_np = rendered.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()
            base_np = rendered_base.detach().cpu().numpy()
            dynamic_np = rendered_dynamic.detach().cpu().numpy()

            bsz, num_cams = rendered_np.shape[:2]
            for b in range(bsz):
                for cam in range(num_cams):
                    stem = f"batch_{batch_idx:05d}_sample_{b}_cam_{cam}"

                    _save_png_chw(gt_np[b, cam], img_dir / f"{stem}_gt.png")
                    _save_png_chw(rendered_np[b, cam], img_dir / f"{stem}_rendered.png")

                    _save_gt_final_panel(
                        gt_np[b, cam],
                        rendered_np[b, cam],
                        gt_final_panel_dir / f"{stem}_panel_gt_final.png",
                    )

                    _save_png_chw(base_np[b, cam], branch_img_dir / f"{stem}_rendered_base.png")
                    _save_png_chw(dynamic_np[b, cam], branch_img_dir / f"{stem}_rendered_dynamic.png")

                    mae = _save_panel(
                        gt_np[b, cam],
                        rendered_np[b, cam],
                        panel_dir / f"{stem}_panel.png",
                        diff_gain=args.diff_gain,
                        diff_gamma=args.diff_gamma,
                    )
                    panel_maes.append(mae)

                    _save_branch_panel(
                        gt_np[b, cam],
                        base_np[b, cam],
                        dynamic_np[b, cam],
                        branch_panel_dir / f"{stem}_panel_base_dynamic.png",
                    )
                    exported += 1

            if args.save_raw_npz:
                payload = {
                    "rendered": rendered_np.astype(np.float16),
                    "gt": gt_np.astype(np.float16),
                    "rendered_base": base_np.astype(np.float16),
                    "rendered_dynamic": dynamic_np.astype(np.float16),
                }
                if "F_residual_t_minus_1" in outputs:
                    payload["residual_mag_t_minus_1"] = torch.linalg.norm(
                        outputs["F_residual_t_minus_1"], dim=2
                    ).detach().cpu().numpy().astype(np.float16)
                if "F_residual_t_plus_1" in outputs:
                    payload["residual_mag_t_plus_1"] = torch.linalg.norm(
                        outputs["F_residual_t_plus_1"], dim=2
                    ).detach().cpu().numpy().astype(np.float16)
                np.savez_compressed(raw_dir / f"batch_{batch_idx:05d}_raw.npz", **payload)

            if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
                break

    print(f"[DONE] exported camera-frames: {exported}")
    print(f"[DONE] images: {img_dir}")
    print(f"[DONE] panels (gt|rendered|diff): {panel_dir}")
    print(f"[DONE] panels (gt|final): {gt_final_panel_dir}")
    print(f"[DONE] branch images: {branch_img_dir}")
    print(f"[DONE] branch panels (gt|base|dynamic): {branch_panel_dir}")
    if panel_maes:
        arr = np.asarray(panel_maes)
        print(f"[DONE] panel MAE mean/min/max: {arr.mean():.6f} / {arr.min():.6f} / {arr.max():.6f}")
    if args.save_raw_npz:
        print(f"[DONE] raw npz: {raw_dir}")


if __name__ == "__main__":
    main()
