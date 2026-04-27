import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare dynamic ghosting between two stage2 inference runs using raw npz outputs.'
    )
    parser.add_argument('--baseline_dir', required=True, type=str,
                        help='baseline inference output dir (contains raw/*.npz or directly *.npz)')
    parser.add_argument('--improved_dir', required=True, type=str,
                        help='improved inference output dir (contains raw/*.npz or directly *.npz)')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='analysis output directory')
    parser.add_argument('--top_k', default=80, type=int,
                        help='number of top improved cases to render panels')
    parser.add_argument('--dynamic_threshold', default=0.8, type=float,
                        help='residual-flow magnitude threshold for dynamic mask')
    parser.add_argument('--dynamic_percentile_fallback', default=85.0, type=float,
                        help='fallback percentile if threshold mask is too sparse')
    parser.add_argument('--min_dynamic_ratio', default=0.005, type=float,
                        help='minimum dynamic mask ratio before fallback to percentile')
    return parser.parse_args()


def resolve_raw_dir(path_str):
    root = Path(path_str)
    raw = root / 'raw'
    if raw.exists() and raw.is_dir():
        return raw
    return root


def list_npz(dir_path):
    files = sorted(dir_path.glob('*.npz'))
    mapping = {}
    for f in files:
        # expected: batch_0000_raw.npz
        name = f.stem
        if 'batch_' in name:
            mapping[name] = f
    return mapping


def chw_to_hwc(img_chw):
    return np.transpose(img_chw, (1, 2, 0))


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def to_u8(img_hwc):
    return (clip01(img_hwc) * 255.0).astype(np.uint8)


def mae(a, b, mask=None):
    err = np.abs(a - b)
    if mask is None:
        return float(err.mean())
    if mask.sum() < 1:
        return float(err.mean())
    if err.ndim == 3:
        mask3 = np.repeat(mask[:, :, None], err.shape[2], axis=2)
    else:
        mask3 = mask
    return float(err[mask3].mean())


def psnr(a, b, mask=None):
    diff = (a - b) ** 2
    if mask is not None and mask.sum() > 0:
        mask3 = np.repeat(mask[:, :, None], diff.shape[2], axis=2)
        mse = float(diff[mask3].mean())
    else:
        mse = float(diff.mean())
    mse = max(mse, 1e-10)
    return float(-10.0 * math.log10(mse))


def error_heatmap(err_map):
    # err_map: HxW, values in [0, +inf)
    vmax = np.percentile(err_map, 99.0)
    vmax = max(vmax, 1e-6)
    x = np.clip(err_map / vmax, 0.0, 1.0)
    # simple red-yellow heatmap without extra deps
    r = x
    g = np.sqrt(x)
    b = np.zeros_like(x)
    return np.stack([r, g, b], axis=-1)


def overlay_mask(img_hwc, mask_hw):
    out = img_hwc.copy()
    # green overlay on dynamic region
    out[mask_hw, 0] *= 0.45
    out[mask_hw, 1] = np.clip(out[mask_hw, 1] * 0.45 + 0.55, 0.0, 1.0)
    out[mask_hw, 2] *= 0.45
    return out


def save_panel(path, gt, old_pred, new_pred, mask):
    # inputs are CHW in [0,1]
    gt_h = clip01(chw_to_hwc(gt))
    old_h = clip01(chw_to_hwc(old_pred))
    new_h = clip01(chw_to_hwc(new_pred))

    old_err = np.abs(old_h - gt_h).mean(axis=2)
    new_err = np.abs(new_h - gt_h).mean(axis=2)

    old_heat = error_heatmap(old_err)
    new_heat = error_heatmap(new_err)
    mask_vis = overlay_mask(gt_h, mask)

    cols = [gt_h, old_h, new_h, old_heat, new_heat, mask_vis]
    h, w, _ = gt_h.shape
    canvas = np.zeros((h, w * len(cols), 3), dtype=np.uint8)
    for i, c in enumerate(cols):
        canvas[:, i * w:(i + 1) * w, :] = to_u8(c)

    Image.fromarray(canvas).save(path)


def dynamic_mask_from_residual(base_npz, new_npz, threshold, min_ratio, pct_fallback):
    # Prefer residual mags from both runs, take union response for fairness.
    cand = []
    for key in ['residual_mag_t_minus_1', 'residual_mag_t_plus_1']:
        if key in base_npz:
            cand.append(base_npz[key])
        if key in new_npz:
            cand.append(new_npz[key])

    if len(cand) == 0:
        return None

    # Each candidate: [B, N, H, W]
    mag = cand[0].astype(np.float32)
    for c in cand[1:]:
        mag = np.maximum(mag, c.astype(np.float32))

    mask = mag > float(threshold)

    # fallback by percentile per frame/cam if mask too sparse
    b, n, h, w = mask.shape
    min_pixels = int(min_ratio * h * w)
    for bi in range(b):
        for ci in range(n):
            if int(mask[bi, ci].sum()) < min_pixels:
                m = mag[bi, ci]
                thr = np.percentile(m, pct_fallback)
                mask[bi, ci] = (m >= thr)

    return mask


def main():
    args = parse_args()

    baseline_raw = resolve_raw_dir(args.baseline_dir)
    improved_raw = resolve_raw_dir(args.improved_dir)

    out_dir = Path(args.output_dir)
    panels_dir = out_dir / 'panels_top_improved'
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    base_files = list_npz(baseline_raw)
    new_files = list_npz(improved_raw)
    common_keys = sorted(set(base_files.keys()) & set(new_files.keys()))

    if len(common_keys) == 0:
        raise RuntimeError('No common raw npz batches found. Run stage2_inference with --save_raw_npz first.')

    rows = []

    for key in common_keys:
        base_npz = np.load(base_files[key], allow_pickle=False)
        new_npz = np.load(new_files[key], allow_pickle=False)

        if 'rendered' not in base_npz or 'rendered' not in new_npz:
            continue

        old_r = base_npz['rendered'].astype(np.float32)  # [B,N,C,H,W]
        new_r = new_npz['rendered'].astype(np.float32)

        if 'gt' in new_npz:
            gt = new_npz['gt'].astype(np.float32)
        elif 'gt' in base_npz:
            gt = base_npz['gt'].astype(np.float32)
        else:
            continue

        dyn_masks = dynamic_mask_from_residual(
            base_npz,
            new_npz,
            threshold=args.dynamic_threshold,
            min_ratio=args.min_dynamic_ratio,
            pct_fallback=args.dynamic_percentile_fallback,
        )

        bsz, ncam = gt.shape[0], gt.shape[1]
        for bi in range(bsz):
            for ci in range(ncam):
                gt_i = gt[bi, ci]
                old_i = old_r[bi, ci]
                new_i = new_r[bi, ci]

                if dyn_masks is not None:
                    mask = dyn_masks[bi, ci]
                else:
                    # fallback: use changed area between old/new render
                    diff_map = np.abs(chw_to_hwc(old_i) - chw_to_hwc(new_i)).mean(axis=2)
                    thr = np.percentile(diff_map, 85.0)
                    mask = diff_map >= thr

                mae_all_old = mae(old_i, gt_i)
                mae_all_new = mae(new_i, gt_i)
                mae_dyn_old = mae(old_i, gt_i, mask)
                mae_dyn_new = mae(new_i, gt_i, mask)

                psnr_all_old = psnr(old_i, gt_i)
                psnr_all_new = psnr(new_i, gt_i)
                psnr_dyn_old = psnr(old_i, gt_i, mask)
                psnr_dyn_new = psnr(new_i, gt_i, mask)

                rows.append({
                    'batch': key,
                    'b': bi,
                    'cam': ci,
                    'dynamic_ratio': float(mask.mean()),
                    'mae_all_old': mae_all_old,
                    'mae_all_new': mae_all_new,
                    'mae_all_improve': mae_all_old - mae_all_new,
                    'mae_dyn_old': mae_dyn_old,
                    'mae_dyn_new': mae_dyn_new,
                    'mae_dyn_improve': mae_dyn_old - mae_dyn_new,
                    'psnr_all_old': psnr_all_old,
                    'psnr_all_new': psnr_all_new,
                    'psnr_all_improve': psnr_all_new - psnr_all_old,
                    'psnr_dyn_old': psnr_dyn_old,
                    'psnr_dyn_new': psnr_dyn_new,
                    'psnr_dyn_improve': psnr_dyn_new - psnr_dyn_old,
                    '_gt': gt_i,
                    '_old': old_i,
                    '_new': new_i,
                    '_mask': mask,
                })

    if len(rows) == 0:
        raise RuntimeError('No comparable samples found in raw npz files.')

    # save csv (without large arrays)
    csv_path = out_dir / 'ghosting_compare_metrics.csv'
    fields = [
        'batch', 'b', 'cam', 'dynamic_ratio',
        'mae_all_old', 'mae_all_new', 'mae_all_improve',
        'mae_dyn_old', 'mae_dyn_new', 'mae_dyn_improve',
        'psnr_all_old', 'psnr_all_new', 'psnr_all_improve',
        'psnr_dyn_old', 'psnr_dyn_new', 'psnr_dyn_improve',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})

    # summary
    def mean_of(k):
        return float(np.mean([r[k] for r in rows]))

    summary_lines = [
        f'samples={len(rows)}',
        f'mae_all_old={mean_of("mae_all_old"):.6f}',
        f'mae_all_new={mean_of("mae_all_new"):.6f}',
        f'mae_all_improve={mean_of("mae_all_improve"):.6f}',
        f'mae_dyn_old={mean_of("mae_dyn_old"):.6f}',
        f'mae_dyn_new={mean_of("mae_dyn_new"):.6f}',
        f'mae_dyn_improve={mean_of("mae_dyn_improve"):.6f}',
        f'psnr_all_old={mean_of("psnr_all_old"):.4f}',
        f'psnr_all_new={mean_of("psnr_all_new"):.4f}',
        f'psnr_all_improve={mean_of("psnr_all_improve"):.4f}',
        f'psnr_dyn_old={mean_of("psnr_dyn_old"):.4f}',
        f'psnr_dyn_new={mean_of("psnr_dyn_new"):.4f}',
        f'psnr_dyn_improve={mean_of("psnr_dyn_improve"):.4f}',
    ]
    (out_dir / 'summary.txt').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')

    # top improved visual panels
    rows_sorted = sorted(rows, key=lambda x: x['mae_dyn_improve'], reverse=True)
    top_k = min(args.top_k, len(rows_sorted))
    for rank in range(top_k):
        r = rows_sorted[rank]
        name = f'{rank:03d}_{r["batch"]}_b{r["b"]}_cam{r["cam"]}_impr_{r["mae_dyn_improve"]:.5f}.png'
        save_panel(
            panels_dir / name,
            r['_gt'],
            r['_old'],
            r['_new'],
            r['_mask'],
        )

    print('=' * 80)
    print('Ghosting comparison finished')
    print(f'Common batches: {len(common_keys)}')
    print(f'Sample count: {len(rows)}')
    print(f'Metrics CSV: {csv_path}')
    print(f'Summary: {out_dir / "summary.txt"}')
    print(f'Top panels: {panels_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()