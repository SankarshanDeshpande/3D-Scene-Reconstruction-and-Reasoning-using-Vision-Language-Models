"""
Standalone script version of the Train_VGGT_3.ipynb cell chain for one scene:
Cell 1 (imports) -> Cell 2 (model) -> Cell 3 (scene setup) -> Cell 5 (inference)
-> Cell 6 (build all_points/all_conf/etc.) -> edited Cell 7 (adaptive-threshold
filtering, with diagnostics) -> Cell 11 (auto-orient + export .ply).

Skips Cells 4, 8, 9, 10 (input grid / render_pointcloud / sweep views / final still)
since those are visual QC only, not required to produce the .ply.

Change SCENE below and rerun for each of the other 4 scenes. Uses FPS-selected frame
indices (from select_frames_fps.py) if available for the scene, falling back to
even-spaced sampling otherwise.
"""

import os, sys, glob, gc
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/mazaveri/hpc-prog/sankarshan/repos/vggt")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

PC_DIR = "/home/mazaveri/hpc-prog/sankarshan/outputs/pointclouds"
os.makedirs(PC_DIR, exist_ok=True)
print("Device:", device, "| dtype:", dtype)

# ============================================================
# ONLY LINE YOU NEED TO CHANGE BETWEEN SCENES
# ============================================================
SCENE = "room2"    # <-- change to "room1", "office0", "office1" for the other runs
# ============================================================

# --- Cell 2: model ---
model = VGGT()
sd = torch.load("/home/mazaveri/hpc-prog/sankarshan/checkpoints/vggt/model.pt", map_location="cpu")
model.load_state_dict(sd, strict=False)
model.eval().to(device)
print("Model loaded")

# --- Cell 3: scene setup ---
scene_root = f"/home/mazaveri/hpc-prog/sankarshan/data/replica/Replica_OCC/sequences/{SCENE}"
color_dir = os.path.join(scene_root, "color")

all_frames = sorted(glob.glob(os.path.join(color_dir, "*.jpg")),
                     key=lambda p: int(os.path.basename(p).split(".")[0]))
N_FRAMES_TOTAL = len(all_frames)
print(f"[{SCENE}] total frames found: {N_FRAMES_TOTAL}")

N_SAMPLES = 128   # back to the original budget -- coverage improvement comes from smarter
                  # frame SELECTION (FPS over GT camera poses), not more frames

FPS_INDEX_PATH = f"/home/mazaveri/hpc-prog/sankarshan/outputs/eval/{SCENE}_fps_indices.txt"
if os.path.exists(FPS_INDEX_PATH):
    with open(FPS_INDEX_PATH) as f:
        idxs = sorted(int(x) for x in f.read().split(","))
    print(f"[{SCENE}] using FPS-selected frame indices ({len(idxs)} frames) from {FPS_INDEX_PATH}")
else:
    idxs = sorted({int(round(i * (N_FRAMES_TOTAL - 1) / (N_SAMPLES - 1))) for i in range(N_SAMPLES)})
    print(f"[{SCENE}] WARNING: no FPS index file found, falling back to even-spaced sampling "
          f"-- run select_frames_fps.py first for {SCENE}")

image_path_list = [all_frames[i] for i in idxs]
print(f"[{SCENE}] using {len(image_path_list)} sampled frames")

LOAD_RES = 1024
VGGT_RES = 518
images, original_coords = load_and_preprocess_images_square(image_path_list, LOAD_RES)
images = images.to(device)
print(tuple(images.shape))

# --- Cell 5+6 rewritten: chunked inference with inter-chunk alignment ---
# BUG FIX: the previous version ran unproject_depth_map_to_point_map() ONCE across
# all chunks' concatenated extrinsics. But each chunk's extrinsics are independently
# anchored (VGGT's camera_head determines pose/scale jointly across whichever frames
# it's given in one forward pass -- with no shared reference between separate calls).
# Concatenating them and unprojecting together silently produced points in 8 different,
# unrelated coordinate systems -- this is exactly the "shattered into disjoint shards"
# result. Confirmed consistent with the depth-eval finding that VGGT's scale varies
# 1.98x-3.02x across independent inference calls (Tables 4.1/4.2).
#
# FIX: make consecutive chunks OVERLAP by a few frames. Each overlapping frame gets
# reconstructed twice (once at the tail of chunk i-1, once at the head of chunk i) --
# use those duplicate reconstructions of the SAME frame as point correspondences to
# solve a similarity transform (rotation + uniform scale + translation, via Umeyama)
# that maps chunk i's coordinate frame onto chunk i-1's (already-aligned) frame.
# Chain this from chunk 0 onward so everything lands in one common frame.

CHUNK_SIZE = 32
OVERLAP = 8   # frames shared between consecutive chunks, used to solve the alignment

def umeyama_alignment(src, dst):
    """Solve for scale, R, t such that: scale * (R @ src.T).T + t ~= dst"""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / len(src)
    scale = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t

def robust_umeyama(src, dst, n_iters=2, keep_frac=0.8):
    """Fit Umeyama, drop the worst-residual correspondences, refit -- guards against
    a few noisy/low-confidence overlap points skewing the alignment."""
    idx = np.arange(len(src))
    for _ in range(n_iters):
        scale, R, t = umeyama_alignment(src[idx], dst[idx])
        pred = scale * (R @ src.T).T + t
        resid = np.linalg.norm(pred - dst, axis=1)
        keep_n = max(10, int(len(idx) * keep_frac))
        idx = np.argsort(resid)[:keep_n]
    return umeyama_alignment(src[idx], dst[idx])

def run_VGGT_one_chunk(model, imr_chunk, dtype):
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            b = imr_chunk[None]
            tokens, ps_idx = model.aggregator(b)
        pose_enc = model.camera_head(tokens)[-1]
        extr, intr = pose_encoding_to_extri_intri(pose_enc, b.shape[-2:])
        dmap, dconf = model.depth_head(tokens, b, ps_idx)
    out = (extr.squeeze(0).cpu().numpy(), intr.squeeze(0).cpu().numpy(),
           dmap.squeeze(0).cpu().numpy(), dconf.squeeze(0).cpu().numpy())
    del tokens, ps_idx, pose_enc, extr, intr, dmap, dconf, b
    torch.cuda.empty_cache()
    return out

imr = F.interpolate(images, size=(VGGT_RES, VGGT_RES), mode="bilinear", align_corners=False)
img_c = imr  # same tensor, reused below for colors
colors_all = img_c.permute(0, 2, 3, 1).cpu().numpy()

oc = original_coords.cpu().numpy()
s = VGGT_RES / LOAD_RES
INSET = 4

n_total_frames = imr.shape[0]
stride = CHUNK_SIZE - OVERLAP
assert stride > 0, "OVERLAP must be smaller than CHUNK_SIZE"

chunk_starts = list(range(0, n_total_frames, stride))
chunk_starts = [cs for cs in chunk_starts if cs < n_total_frames]
# make sure the final chunk reaches the last frame
if chunk_starts[-1] + CHUNK_SIZE < n_total_frames:
    chunk_starts.append(n_total_frames - CHUNK_SIZE)

aligned_points_list, aligned_conf_list, aligned_valid_list, aligned_colors_list = [], [], [], []
prev_points_aligned = None
prev_valid_mask = None
prev_conf = None
prev_chunk_range = None

# --- Checkpointing: save each chunk's aligned result as soon as it's computed, and
# skip straight past already-completed chunks on a rerun. This means a kill partway
# through no longer throws away earlier chunks' (expensive) GPU work.
CACHE_DIR = f"{PC_DIR}/../eval/{SCENE}/chunk_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

for ci, start in enumerate(chunk_starts):
    end = min(start + CHUNK_SIZE, n_total_frames)
    cache_path = os.path.join(CACHE_DIR, f"chunk_{ci:03d}.npz")
    expected_frame_idxs = np.array(idxs[start:end])   # GLOBAL frame indices this chunk should cover

    cache_valid = False
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if "frame_idxs" in cached.files and np.array_equal(cached["frame_idxs"], expected_frame_idxs):
            cache_valid = True
        else:
            print(f"[{SCENE}] chunk {start}:{end} cache STALE (built from a different frame "
                  f"selection) -- ignoring and recomputing")

    if cache_valid:
        points_aligned = cached["points_aligned"]
        valid_chunk = cached["valid_chunk"]
        dconf = cached["conf"]
        print(f"[{SCENE}] chunk {start}:{end} loaded from cache ({cache_path})")
    else:
        chunk_imr = imr[start:end]
        extr, intr, dmap, dconf = run_VGGT_one_chunk(model, chunk_imr, dtype)
        points_chunk = unproject_depth_map_to_point_map(dmap, extr, intr)  # chunk-local coords, shape (n_chunk, H, W, 3)

        # build this chunk's valid mask (padding) -- same geometry logic as before, per-frame
        n_chunk = points_chunk.shape[0]
        valid_chunk = np.zeros((n_chunk, VGGT_RES, VGGT_RES), dtype=bool)
        for j in range(n_chunk):
            gi = start + j  # global frame index in the 0..n_total_frames-1 sample
            x1, y1, x2, y2 = oc[gi, :4] * s
            r0, r1 = int(np.ceil(y1)) + INSET, int(np.floor(y2)) - INSET
            c0, c1 = int(np.ceil(x1)) + INSET, int(np.floor(x2)) - INSET
            valid_chunk[j, max(r0, 0):r1, max(c0, 0):c1] = True

        if ci == 0:
            # first chunk defines the global reference frame -- no alignment needed
            points_aligned = points_chunk
        else:
            # overlap frames: the first OVERLAP frames of this chunk are the same global
            # frames as the last OVERLAP frames of the previous chunk
            n_overlap_actual = min(OVERLAP, n_chunk, prev_chunk_range[1] - prev_chunk_range[0])
            src_pts_list, dst_pts_list = [], []
            for k in range(n_overlap_actual):
                this_local = k                              # index within this chunk
                prev_local = (prev_chunk_range[1] - prev_chunk_range[0]) - n_overlap_actual + k  # index within prev chunk
                m = valid_chunk[this_local] & (dconf[this_local] > 1.0) & prev_valid_mask[prev_local] & (prev_conf[prev_local] > 1.0)
                if m.sum() < 20:
                    continue
                src_pts_list.append(points_chunk[this_local][m])
                dst_pts_list.append(prev_points_aligned[prev_local][m])
            if not src_pts_list:
                print(f"[{SCENE}] WARNING: chunk {ci} has no usable overlap correspondences -- "
                      f"leaving unaligned (increase OVERLAP or lower confidence threshold above)")
                points_aligned = points_chunk
            else:
                src = np.concatenate(src_pts_list, axis=0)
                dst = np.concatenate(dst_pts_list, axis=0)
                # subsample for speed if huge
                if len(src) > 20000:
                    pick = np.random.choice(len(src), 20000, replace=False)
                    src, dst = src[pick], dst[pick]
                scale, R, t = robust_umeyama(src, dst)
                flat = points_chunk.reshape(-1, 3)
                aligned_flat = scale * (R @ flat.T).T + t
                points_aligned = aligned_flat.reshape(points_chunk.shape)
                resid = np.linalg.norm((scale * (R @ src.T).T + t) - dst, axis=1)
                print(f"[{SCENE}] chunk {ci} aligned via {len(src)} overlap points "
                      f"| scale={scale:.3f} | mean residual={resid.mean():.4f} m")

        np.savez(cache_path, points_aligned=points_aligned, valid_chunk=valid_chunk, conf=dconf,
                 frame_idxs=expected_frame_idxs)
        print(f"[{SCENE}] chunk {start}:{end} done, cached | GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    aligned_points_list.append(points_aligned)
    aligned_conf_list.append(dconf)
    aligned_valid_list.append(valid_chunk)
    aligned_colors_list.append(colors_all[start:end])

    prev_points_aligned = points_aligned
    prev_valid_mask = valid_chunk
    prev_conf = dconf
    prev_chunk_range = (start, end)

points_3d_full = np.concatenate(aligned_points_list, axis=0)
conf_full = np.concatenate(aligned_conf_list, axis=0)
valid_full = np.concatenate(aligned_valid_list, axis=0)
colors_full = np.concatenate(aligned_colors_list, axis=0)

print(f"[{SCENE}] padding discards {100*(1-valid_full.mean()):.1f}% of pixels")

all_points = points_3d_full.reshape(-1, 3)
all_colors = np.clip(colors_full.reshape(-1, 3), 0, 1)
all_conf = conf_full.reshape(-1)
all_valid = valid_full.reshape(-1)
print(f"[{SCENE}] conf percentiles:", np.round(np.percentile(all_conf, [50, 75, 90, 95, 99]), 2))

# --- Edited Cell 7: filtering, with diagnostics + Fix 1 (adaptive conf) + Fix 2 (percentile SOR) ---
n_total = len(all_valid)
print(f"[{SCENE}] total candidate points: {n_total:,}")

CONF_THRES = 3.0   # reverted to original fixed value -- isolating frame count as the only change this run
print(f"[{SCENE}] CONF_THRES: {CONF_THRES}")

keep = all_valid & (all_conf > CONF_THRES) & np.isfinite(all_points).all(axis=1)
pts, cols_ = all_points[keep], all_colors[keep]
print(f"[{SCENE}] after confidence filter: {keep.sum():,} / {n_total:,} ({100*keep.sum()/n_total:.1f}%)")

d = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
inl = d < np.percentile(d, 99.0)
n_before_bbox = len(pts)
pts, cols_ = pts[inl], cols_[inl]
print(f"[{SCENE}] after bounding-outlier filter: {len(pts):,} / {n_before_bbox:,} kept")

from scipy.spatial import cKDTree
tree = cKDTree(pts)
dist, _ = tree.query(pts, k=9, workers=-1)
mdist = dist[:, 1:].mean(axis=1)

n_before_sor = len(pts)
sor = mdist < mdist.mean() + 2.0 * mdist.std()   # reverted to original global-mean SOR
pts, cols_ = pts[sor], cols_[sor]
print(f"[{SCENE}] after statistical outlier removal: {len(pts):,} / {n_before_sor:,} kept")

print(f"[{SCENE}] FINAL points kept: {len(pts):,}")

# --- Cell 11: auto-orient + export ---
import trimesh
import open3d as o3d

def auto_orient_points(pts_in, up_axis="Y"):
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts_in)
    plane_model, inliers = pcd_tmp.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=2000)
    a, b, c, d_ = plane_model
    normal = np.array([a, b, c]); normal = normal / np.linalg.norm(normal)
    target = np.array([0, 1, 0]) if up_axis == "Y" else np.array([0, 0, 1])
    centroid = pts_in.mean(axis=0)
    floor_point = pts_in[inliers[0]]
    if np.dot(centroid - floor_point, normal) < 0:
        normal = -normal
    v = np.cross(normal, target); s_ = np.linalg.norm(v); cos_a = np.dot(normal, target)
    if s_ < 1e-8:
        R = np.eye(3) if cos_a > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cos_a) / (s_ ** 2))
    return pts_in @ R.T

pts = auto_orient_points(pts, up_axis="Y")

out_path = os.path.join(PC_DIR, f"{SCENE}.ply")
trimesh.PointCloud(vertices=pts, colors=(cols_ * 255).astype(np.uint8)).export(out_path)
print(f"[{SCENE}] saved {len(pts):,} points to {out_path}")

# --- Cell 12: cleanup ---
del images, imr, colors_all, aligned_points_list, aligned_conf_list, aligned_valid_list, aligned_colors_list, all_points, all_colors, all_conf, all_valid, pts, cols_
torch.cuda.empty_cache()
gc.collect()
print(f"[{SCENE}] done.")