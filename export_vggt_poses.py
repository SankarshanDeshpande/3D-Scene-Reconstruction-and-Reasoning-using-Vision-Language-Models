"""
export_vggt_poses.py

Re-runs VGGT inference for a given scene and saves the predicted camera-to-world
poses as per-frame .txt files that generate_gsa_results.py can consume directly
via ScannetDataset.

The output directory pose_vggt/ is a drop-in replacement for the GT pose/
directory: same filename convention (<frame_index>.txt), same 4x4 row-major
space-separated format that np.loadtxt() reads.

For frames that fall between VGGT's 128-sample indices, the nearest sampled
frame's pose is used (nearest-neighbour in frame index space).

Usage (run inside the vlm conda env on a GPU node):
    conda activate vlm
    python scripts/export_vggt_poses.py --scene room0
"""

import argparse
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

# -- args ----------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--scene", required=True,
               choices=["room0", "room1", "room2", "office0", "office1"])
p.add_argument("--replica_root",
               default="/home/mazaveri/hpc-prog/sankarshan/data/replica/Replica_OCC/sequences")
p.add_argument("--vggt_repo",
               default="/home/mazaveri/hpc-prog/sankarshan/repos/vggt")
p.add_argument("--checkpoint",
               default="/home/mazaveri/hpc-prog/sankarshan/checkpoints/vggt/model.pt")
p.add_argument("--n_samples", type=int, default=128,
               help="Frames passed to VGGT (must match your original reconstruction run)")
p.add_argument("--gsa_stride", type=int, default=5,
               help="Stride used in generate_gsa_results.py (default 5)")
p.add_argument("--img_size", type=int, default=518,
               help="VGGT fixed resolution (default 518, matches vggt_fixed_resolution in your notebook)")
args = p.parse_args()

# -- paths ---------------------------------------------------------------------
scene_dir = Path(args.replica_root) / args.scene
color_dir = scene_dir / "color"
out_dir   = scene_dir / "pose_vggt"
out_dir.mkdir(exist_ok=True)

# -- frame index sets ----------------------------------------------------------
all_frame_files = sorted(color_dir.glob("*.jpg"), key=lambda f: int(f.stem))
total_frames    = len(all_frame_files)
print(f"Scene: {args.scene}  |  total frames in color/: {total_frames}")

# 128 uniform indices — identical logic to your reconstruction notebook
vggt_indices = np.linspace(0, total_frames - 1, args.n_samples, dtype=int).tolist()
print(f"VGGT will process {len(vggt_indices)} frames "
      f"(indices {vggt_indices[0]} … {vggt_indices[-1]})")

# stride-5 indices that generate_gsa_results.py will request
gsa_indices = list(range(0, total_frames, args.gsa_stride))
print(f"GSA will request  {len(gsa_indices)} frames (stride={args.gsa_stride})")

# -- load VGGT model -----------------------------------------------------------
sys.path.insert(0, args.vggt_repo)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.bfloat16   # same as your notebook (autocast bfloat16)
print(f"Device: {device}")

print("Loading VGGT checkpoint …")
model = VGGT.from_pretrained(args.checkpoint)
model = model.to(device).eval()
print("Checkpoint loaded.")

# -- VGGT inference ------------------------------------------------------------
image_paths = [str(all_frame_files[i]) for i in vggt_indices]

print(f"Preprocessing {len(image_paths)} images (img_size={args.img_size}) …")
images = load_and_preprocess_images(image_paths, img_size=args.img_size)
# shape: (1, N, C, H, W) — same as your notebook
images = images.to(device)

print("Running VGGT forward pass …")
with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
    predictions = model(images)

# predictions["extrinsic"]: (1, N, 3, 4) — camera-to-world, rows 0-2 of the 4x4 matrix
# This matches the VGGT API exactly; the bottom row [0,0,0,1] is implicit.
extrinsics_34 = predictions["extrinsic"][0].cpu().float().numpy()  # (N, 3, 4)
print(f"Extrinsics extracted: shape {extrinsics_34.shape}")

# Sanity check: frame 0 should be close to identity (VGGT anchors to first frame)
print("Frame-0 rotation block (expect ~identity):")
print(extrinsics_34[0, :3, :3].round(4))

# -- build 4x4 c2w matrices and index by scene frame number -------------------
def to_4x4(mat_34: np.ndarray) -> np.ndarray:
    """Pad (3,4) ? (4,4) by appending [0,0,0,1]."""
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
    return np.vstack([mat_34, bottom])

# Map: scene_frame_index ? 4x4 c2w matrix
vggt_pose_at_frame = {
    int(vggt_indices[i]): to_4x4(extrinsics_34[i])
    for i in range(len(vggt_indices))
}
sampled_sorted = sorted(vggt_pose_at_frame.keys())

def nearest_pose(query_idx: int) -> np.ndarray:
    """Return the VGGT pose from the nearest sampled frame (by frame index)."""
    dists   = np.abs(np.array(sampled_sorted) - query_idx)
    nearest = sampled_sorted[int(np.argmin(dists))]
    return vggt_pose_at_frame[nearest]

# -- write pose_vggt/<frame_index>.txt -----------------------------------------
# Format: 4x4 matrix, 4 rows of 4 space-separated floats, 10 decimal places.
# This is exactly what ScannetDataset reads via:
#     pose = np.loadtxt(pose_path).reshape(4, 4)
written = 0
for frame_idx in gsa_indices:
    c2w      = nearest_pose(frame_idx)
    out_path = out_dir / f"{frame_idx}.txt"
    np.savetxt(str(out_path), c2w, fmt="%.10f")
    written += 1

print(f"\nWrote {written} pose files ? {out_dir}/")
print("Quick verification:")
print(f"  File count   : {written}  (expected {len(gsa_indices)})")
print(f"  First file   : {out_dir}/{gsa_indices[0]}.txt")
print(f"  Last file    : {out_dir}/{gsa_indices[-1]}.txt")

# Print the first pose as a sanity check
first_c2w = np.loadtxt(str(out_dir / f"{gsa_indices[0]}.txt"))
print(f"\nPose for frame {gsa_indices[0]} (should be ~identity for frame 0):")
print(first_c2w.round(4))
print("\nDone.")