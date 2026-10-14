"""
Quantitative evaluation of VGGT-predicted depth against ReplicaOcc ground-truth depth,
for the five existing reconstructions (room0, room1, room2, office0, office1).

WHAT THIS DOES
    1. For each scene, selects the same 128 evenly-spaced frames the project's reconstruction
       pipeline is documented to have used.
    2. Runs VGGT-1B inference on those frames (same checkpoint / same preprocessing settings
       documented in the project: img_load_resolution=1024, vggt_fixed_resolution=518) and saves
       the raw per-frame predicted depth to disk (this was NOT previously saved anywhere in the
       project outputs -- see chat notes).
    3. Loads the corresponding ReplicaOcc ground-truth depth, converts with the CONFIRMED correct
       scale factor (png_depth_scale = 6553.5, NOT /1000).
    4. Undoes VGGT's pad-then-resize-to-square preprocessing so predicted depth aligns pixel-for-
       pixel with the original 1200x680 GT depth.
    5. Applies a valid-pixel mask (GT>0 & finite, pred finite & >0). No confidence filtering is
       applied here -- metrics must come from the raw predictions per project instructions.
    6. Reports BOTH:
         - RAW comparison (no scale correction) -- meaningful only if VGGT depth is claimed metric.
         - SCALE-ALIGNED comparison, using the standard per-frame median-ratio alignment
           (matches how the VGGT paper itself reports DTU/ETH3D numbers, and matches the protocol
           already set up but never completed in this project -- see project_context.md Sec 4.4).
       VGGT's depth output is scale-ambiguous by construction (monocular/multi-view feed-forward
       depth without a metric prior beyond what limited camera-pose supervision gives it), so the
       RAW numbers are reported for transparency but the ALIGNED numbers are the ones that should
       go in the dissertation.
    7. Computes MAE, RMSE, AbsRel, delta<1.25, delta<1.25^2, delta<1.25^3 per scene, and an
       AGGREGATE computed over the union of all valid pixels across all scenes (not a mean of
       per-scene means) -- plus, separately, the simple mean-of-scenes for comparison.
    8. Saves per-scene and aggregate results to CSV/JSON.
    9. Generates RGB | GT depth | Pred depth | Error map visualizations for a few frames/scene.

WHAT THIS DOES NOT DO
    - No Chamfer distance / point-cloud metric (see eval_pointcloud_pose.py stub + README below --
      requires a camera-convention sanity check first).
    - No pose/ATE evaluation (same reason).
    - No ConceptGraphs / scene-graph / SceneLLM evaluation (future pipeline stages, out of scope).
    - No CO3D training-loss metrics (that's a different, earlier experiment; not included).

BEFORE RUNNING: confirm the paths in CONFIG below against your actual HPC layout (see the
inspection commands provided alongside this script). This script reconstructs VGGT's standard
public inference call pattern; if your project's actual inference cell differs, swap it in but
keep the depth-saving logic (Part 2) unchanged.
"""

import os
import json
import csv
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG -- edit these to match your confirmed HPC paths
# ============================================================================
BASE = "/home/mazaveri/hpc-prog/sankarshan"
DATA_ROOT = f"{BASE}/data/replica/Replica_OCC/sequences"
CHECKPOINT_PATH = f"{BASE}/checkpoints/vggt/model.pt"
OUT_DIR = f"{BASE}/outputs/eval"

SCENES = ["room0", "room1", "room2", "office0", "office1"]
NUM_FRAMES = 128                      # per project docs (deviations.md #13, project_context.md 4.5)
DEPTH_SCALE = 6553.5                  # CONFIRMED correct scale (NOT 1000.0) -- deviations.md #19
IMG_LOAD_RESOLUTION = 1024
VGGT_FIXED_RESOLUTION = 518
NUM_QUALITATIVE_FRAMES_PER_SCENE = 3  # for the RGB|GT|Pred|Error visualization

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# PART 1 -- I/O helpers, built strictly from the documented ReplicaOcc layout
# ============================================================================

def list_scene_frames(scene):
    """Return sorted list of frame indices (int) available for a scene, based on color/ dir."""
    color_dir = os.path.join(DATA_ROOT, scene, "color")
    files = [f for f in os.listdir(color_dir) if f.lower().endswith(".jpg")]
    idxs = sorted(int(os.path.splitext(f)[0]) for f in files)
    return idxs


def select_evenly_spaced(idxs, n):
    """
    Select n evenly-spaced indices from the available frame list.
    NOTE: this matches the project's documented 'evenly-spaced' sampling convention
    (deviations.md #13), but the EXACT indices used to build the existing {scene}.ply files
    are not confirmed anywhere in the project docs. This does not invalidate depth evaluation
    (each frame is independently scored against its own GT), but for perfect consistency with
    the existing reconstructions, confirm the sampling method used in your actual script.
    """
    if len(idxs) <= n:
        return idxs
    positions = np.linspace(0, len(idxs) - 1, n).round().astype(int)
    return [idxs[p] for p in positions]


def load_intrinsics(scene):
    """
    Parse intrinsic/intrinsic_color.txt. Handles either a 4x4 or 3x3 whitespace-separated matrix.
    Only room0's intrinsics are confirmed in the project docs (fx=fy=600, cx=599.5, cy=339.5,
    1200x680) -- this function reads per-scene so it will pick up any differences automatically.
    Verify the parsed values against your `cat intrinsic_color.txt` output before trusting them.
    """
    path = os.path.join(DATA_ROOT, scene, "intrinsic", "intrinsic_color.txt")
    vals = np.loadtxt(path).flatten()
    if vals.size >= 16:
        K = vals[:16].reshape(4, 4)[:3, :3]
    elif vals.size >= 9:
        K = vals[:9].reshape(3, 3)
    else:
        raise ValueError(f"Unexpected intrinsic file format for {scene}: {path}")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return dict(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))


def load_gt_depth(scene, frame_idx):
    """Load ReplicaOcc GT depth PNG (uint16) and convert with the CONFIRMED correct scale."""
    path = os.path.join(DATA_ROOT, scene, "depth", f"{frame_idx}.png")
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    depth_m = raw.astype(np.float32) / DEPTH_SCALE  # metres
    return depth_m


def load_rgb(scene, frame_idx):
    path = os.path.join(DATA_ROOT, scene, "color", f"{frame_idx}.jpg")
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img


# ============================================================================
# PART 2 -- VGGT inference + depth saving
#   Standard VGGT public API (aggregator -> camera_head / depth_head), matching the
#   img_load_resolution / vggt_fixed_resolution settings documented for this project.
#   >>> If your actual notebook's inference cell differs, replace this function body but
#   >>> keep saving depth_pred_native / conf_native exactly as done here. <<<
# ============================================================================

_model = None
_dtype = None

def get_model():
    """
    Loads VGGT-1B exactly as the notebook does: VGGT() + load_state_dict(strict=False) via
    checkpoints/vggt/model.pt. dtype selection (bf16 on Ampere+/Hopper, else fp16) mirrors the
    standard VGGT demo pattern -- the notebook's `dtype` variable wasn't captured in the grep,
    so confirm this matches your notebook's actual dtype cell before trusting results at scale;
    it will not silently break anything if it doesn't (autocast just runs at a different precision).
    """
    global _model, _dtype
    if _model is None:
        from vggt.models.vggt import VGGT
        _model = VGGT()
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        _model.load_state_dict(state_dict, strict=False)  # matches project's documented load pattern
        _model.eval().to(DEVICE)
        if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            _dtype = torch.bfloat16
        else:
            _dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    return _model, _dtype


def run_VGGT(model, images, dtype, resolution=VGGT_FIXED_RESOLUTION):
    """
    Exact port of the notebook's run_VGGT() (Train_VGGT_1.ipynb, cell after mask loading):
    images arrive at img_load_resolution (1024, padded-square) and are downsized here to
    `resolution` (518) via bilinear interpolation before the forward pass -- this two-stage
    resize (1024 -> 518) is why original_coords (computed in 1024-space by
    load_and_preprocess_images_square) must be RESCALED before use, not used as-is.
    """
    import torch.nn.functional as F
    assert len(images.shape) == 4 and images.shape[1] == 3
    images_resized = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batched = images_resized[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images_batched)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batched.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batched, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def unpad_to_native(proc_array, orig_coord, proc_resolution):
    """
    Inverts the pad-then-resize-to-square preprocessing using the REAL original_coords returned
    by load_and_preprocess_images_square (x1,y1,x2,y2,width,height, in img_load_resolution=1024
    space), rescaled to the model's actual processing resolution (518), since run_VGGT further
    downsizes 1024->518 before the forward pass. This mirrors what the notebook's
    load_and_preprocess_mask_square does for masks, applied here to a single-channel array
    (depth or confidence) instead.
    """
    x1, y1, x2, y2, width, height = orig_coord
    scale = proc_resolution / IMG_LOAD_RESOLUTION
    x1s, y1s, x2s, y2s = x1 * scale, y1 * scale, x2 * scale, y2 * scale
    x1i, y1i = int(round(x1s)), int(round(y1s))
    x2i, y2i = max(x1i + 1, int(round(x2s))), max(y1i + 1, int(round(y2s)))
    crop = proc_array[y1i:y2i, x1i:x2i]
    native = cv2.resize(crop, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    return native


def run_vggt_and_cache_depth(scene, frame_idxs):
    """
    Runs VGGT on ALL of a scene's (missing/uncached) frames in a SINGLE batched call, exactly as
    the notebook does (proven safe for 128 frames on shard:20 per project comments), then saves
    per-frame native-resolution predicted depth + confidence to
    OUT_DIR/{scene}/depth_pred/{frame_idx}.npy. Returns dict frame_idx -> path to cached .npy.
    """
    from vggt.utils.load_fn import load_and_preprocess_images_square

    cache_dir = os.path.join(OUT_DIR, scene, "depth_pred")
    os.makedirs(cache_dir, exist_ok=True)

    cached = {}
    missing = []
    for fi in frame_idxs:
        p = os.path.join(cache_dir, f"{fi}.npy")
        if os.path.exists(p):
            cached[fi] = p
        else:
            missing.append(fi)

    if missing:
        model, dtype = get_model()
        image_paths = [os.path.join(DATA_ROOT, scene, "color", f"{fi}.jpg") for fi in missing]

        images, original_coords = load_and_preprocess_images_square(image_paths, IMG_LOAD_RESOLUTION)
        images = images.to(DEVICE)
        original_coords = original_coords.numpy()  # (N, 5): x1,y1,x2,y2,width,height in 1024-space

        extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, VGGT_FIXED_RESOLUTION)
        # depth_map / depth_conf shape: (N, H_proc, W_proc, 1) or (N, H_proc, W_proc) -- squeeze defensively
        depth_map = np.squeeze(depth_map)
        depth_conf = np.squeeze(depth_conf)
        if depth_map.ndim == 2:  # single-frame batch collapsed an axis
            depth_map = depth_map[None]
            depth_conf = depth_conf[None]

        for i, fi in enumerate(missing):
            d_native = unpad_to_native(depth_map[i], original_coords[i], VGGT_FIXED_RESOLUTION)
            c_native = unpad_to_native(depth_conf[i], original_coords[i], VGGT_FIXED_RESOLUTION)
            out_path = os.path.join(cache_dir, f"{fi}.npy")
            np.save(out_path, np.stack([d_native, c_native], axis=0))
            cached[fi] = out_path

    return cached


# ============================================================================
# PART 3 -- Metrics
# ============================================================================

def compute_metrics(pred, gt, valid_mask):
    pred_v = pred[valid_mask]
    gt_v = gt[valid_mask]
    mae = float(np.mean(np.abs(pred_v - gt_v)))
    rmse = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))
    abs_rel = float(np.mean(np.abs(pred_v - gt_v) / gt_v))
    ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < 1.25 ** 2))
    d3 = float(np.mean(ratio < 1.25 ** 3))
    return dict(MAE=mae, RMSE=rmse, AbsRel=abs_rel, delta1=d1, delta2=d2, delta3=d3,
                n_valid_pixels=int(valid_mask.sum()))


def median_ratio_scale(pred, gt, valid_mask):
    """Standard per-frame median-ratio scale alignment (matches VGGT paper's own DTU/ETH3D
    evaluation protocol, and the protocol this project had already set up -- project_context.md 4.4)."""
    scale = np.median(gt[valid_mask] / pred[valid_mask])
    return pred * scale, float(scale)


# ============================================================================
# PART 4 -- Main evaluation loop
# ============================================================================

def evaluate_scene(scene):
    idxs_all = list_scene_frames(scene)
    frame_idxs = select_evenly_spaced(idxs_all, NUM_FRAMES)
    intr = load_intrinsics(scene)

    depth_cache = run_vggt_and_cache_depth(scene, frame_idxs)

    raw_valid_masks, raw_preds, raw_gts = [], [], []
    aligned_preds, per_frame_scales = [], []
    n_pixels_total = 0

    for fi in frame_idxs:
        arr = np.load(depth_cache[fi])
        pred, conf = arr[0], arr[1]
        gt = load_gt_depth(scene, fi)

        if pred.shape != gt.shape:
            gt_resized = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            gt_resized = gt

        valid = (gt_resized > 0) & np.isfinite(gt_resized) & np.isfinite(pred) & (pred > 0)

        n_pixels_total += valid.sum()
        raw_valid_masks.append(valid)
        raw_preds.append(pred)
        raw_gts.append(gt_resized)

        pred_aligned, scale = median_ratio_scale(pred, gt_resized, valid) if valid.sum() > 0 else (pred, np.nan)
        aligned_preds.append(pred_aligned)
        per_frame_scales.append(scale)

    # Aggregate over the UNION of all valid pixels in the scene (not mean-of-frame-means)
    pred_cat = np.concatenate([p[m] for p, m in zip(raw_preds, raw_valid_masks)])
    gt_cat = np.concatenate([g[m] for g, m in zip(raw_gts, raw_valid_masks)])
    pred_aligned_cat = np.concatenate([p[m] for p, m in zip(aligned_preds, raw_valid_masks)])

    full_mask = np.ones_like(pred_cat, dtype=bool)
    raw_metrics = compute_metrics(pred_cat, gt_cat, full_mask)
    aligned_metrics = compute_metrics(pred_aligned_cat, gt_cat, full_mask)

    result = dict(
        scene=scene,
        n_frames=len(frame_idxs),
        n_valid_pixels=int(n_pixels_total),
        median_scale_factor=float(np.nanmedian(per_frame_scales)),
        raw=raw_metrics,
        scale_aligned=aligned_metrics,
    )

    # Qualitative visualizations
    save_qualitative(scene, frame_idxs[:NUM_QUALITATIVE_FRAMES_PER_SCENE],
                      raw_preds[:NUM_QUALITATIVE_FRAMES_PER_SCENE],
                      aligned_preds[:NUM_QUALITATIVE_FRAMES_PER_SCENE],
                      raw_gts[:NUM_QUALITATIVE_FRAMES_PER_SCENE])

    return result


def save_qualitative(scene, frame_idxs, preds, preds_aligned, gts):
    out_dir = os.path.join(OUT_DIR, scene)
    os.makedirs(out_dir, exist_ok=True)
    for fi, pred, pred_al, gt in zip(frame_idxs, preds, preds_aligned, gts):
        rgb = load_rgb(scene, fi)
        err = np.abs(pred_al - gt)
        err[gt <= 0] = np.nan

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(rgb); axes[0].set_title(f"{scene} frame {fi} - RGB"); axes[0].axis("off")
        im1 = axes[1].imshow(gt, cmap="viridis", vmin=0, vmax=np.nanpercentile(gt, 95))
        axes[1].set_title("GT depth (m)"); axes[1].axis("off"); fig.colorbar(im1, ax=axes[1], fraction=0.046)
        im2 = axes[2].imshow(pred_al, cmap="viridis", vmin=0, vmax=np.nanpercentile(gt, 95))
        axes[2].set_title("VGGT depth, scale-aligned (m)"); axes[2].axis("off"); fig.colorbar(im2, ax=axes[2], fraction=0.046)
        im3 = axes[3].imshow(err, cmap="hot", vmin=0, vmax=np.nanpercentile(err, 95))
        axes[3].set_title("|Error| (m)"); axes[3].axis("off"); fig.colorbar(im3, ax=axes[3], fraction=0.046)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"qual_frame{fi}.png"), dpi=150)
        plt.close(fig)


def main():
    all_results = []
    for scene in SCENES:
        print(f"=== Evaluating {scene} ===")
        r = evaluate_scene(scene)
        all_results.append(r)
        print(json.dumps(r, indent=2))

    # Aggregate across all 5 scenes (union of valid pixels, not mean of scene means)
    total_valid = sum(r["n_valid_pixels"] for r in all_results)

    def weighted_mean(key_path):
        num = sum(r["scale_aligned"][key_path] * r["n_valid_pixels"] for r in all_results)
        return num / total_valid

    aggregate_pixel_weighted = {
        k: weighted_mean(k) for k in ["MAE", "RMSE", "AbsRel", "delta1", "delta2", "delta3"]
    }
    aggregate_simple_mean = {
        k: float(np.mean([r["scale_aligned"][k] for r in all_results]))
        for k in ["MAE", "RMSE", "AbsRel", "delta1", "delta2", "delta3"]
    }

    summary = dict(
        per_scene=all_results,
        aggregate_pixel_weighted=aggregate_pixel_weighted,  # PREFERRED aggregation
        aggregate_simple_mean_of_scenes=aggregate_simple_mean,  # reported for reference only
        total_frames=sum(r["n_frames"] for r in all_results),
        total_valid_pixels=total_valid,
        depth_scale_used=DEPTH_SCALE,
        note="scale_aligned metrics use per-frame median-ratio alignment; raw metrics also saved per scene for transparency.",
    )

    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUT_DIR, "results_table.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Scene", "Frames", "ValidPixels", "MAE", "RMSE", "AbsRel", "delta1", "delta2", "delta3"])
        for r in all_results:
            m = r["scale_aligned"]
            writer.writerow([r["scene"], r["n_frames"], r["n_valid_pixels"],
                              f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['AbsRel']:.4f}",
                              f"{m['delta1']:.4f}", f"{m['delta2']:.4f}", f"{m['delta3']:.4f}"])
        m = aggregate_pixel_weighted
        writer.writerow(["Mean (pixel-weighted)", summary["total_frames"], total_valid,
                          f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['AbsRel']:.4f}",
                          f"{m['delta1']:.4f}", f"{m['delta2']:.4f}", f"{m['delta3']:.4f}"])

    print("\nSaved:", os.path.join(OUT_DIR, "results_summary.json"))
    print("Saved:", os.path.join(OUT_DIR, "results_table.csv"))


if __name__ == "__main__":
    main()
