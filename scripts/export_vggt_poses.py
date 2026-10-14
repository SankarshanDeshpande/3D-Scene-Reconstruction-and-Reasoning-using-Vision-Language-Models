import argparse, sys
import numpy as np
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument("--scene", required=True,
               choices=["room0","room1","room2","office0","office1"])
p.add_argument("--replica_root",
               default="/home/mazaveri/hpc-prog/sankarshan/data/replica/Replica_OCC/sequences")
p.add_argument("--vggt_repo",
               default="/home/mazaveri/hpc-prog/sankarshan/repos/vggt")
p.add_argument("--checkpoint",
               default="/home/mazaveri/hpc-prog/sankarshan/checkpoints/vggt/model.pt")
p.add_argument("--n_samples", type=int, default=128)
p.add_argument("--gsa_stride", type=int, default=5)
p.add_argument("--img_size", type=int, default=518)
args = p.parse_args()

scene_dir = Path(args.replica_root) / args.scene
color_dir = scene_dir / "color"
out_dir   = scene_dir / "pose_vggt"
out_dir.mkdir(exist_ok=True)

all_frames   = sorted(color_dir.glob("*.jpg"), key=lambda f: int(f.stem))
total        = len(all_frames)
vggt_indices = np.linspace(0, total - 1, args.n_samples, dtype=int).tolist()
gsa_indices  = list(range(0, total, args.gsa_stride))
all_indices  = list(range(0, total))

print(f"Scene        : {args.scene}")
print(f"Total frames : {total}")
print(f"VGGT frames  : {len(vggt_indices)}  ({vggt_indices[0]} to {vggt_indices[-1]})")
print(f"GSA frames   : {len(gsa_indices)}  (stride={args.gsa_stride})")

sys.path.insert(0, args.vggt_repo)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device       : {device}")

print("Loading VGGT checkpoint ...")
ckpt = torch.load(args.checkpoint, map_location=device)
model = VGGT()
model.load_state_dict(ckpt)
model = model.to(device).eval()
print("Checkpoint loaded.")

image_paths = [str(all_frames[i]) for i in vggt_indices]
print(f"Preprocessing {len(image_paths)} images ...")
images = load_and_preprocess_images(image_paths).to(device)

print("Running VGGT forward pass ...")
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    predictions = model(images)

extrinsics, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], build_intrinsics=False)
extrinsics_34 = extrinsics[0].cpu().float().numpy()  # (N, 3, 4)
print(f"Extrinsics shape : {extrinsics_34.shape}")
print("Frame-0 rotation (expect ~identity):")
print(extrinsics_34[0, :3, :3].round(4))

def to_4x4(m34):
    return np.vstack([m34, [0.0, 0.0, 0.0, 1.0]])

pose_at = {int(vggt_indices[i]): to_4x4(extrinsics_34[i])
           for i in range(len(vggt_indices))}
sampled = sorted(pose_at.keys())

def nearest(idx):
    dists = np.abs(np.array(sampled) - idx)
    return pose_at[sampled[int(np.argmin(dists))]]

for frame_idx in all_indices:
    np.savetxt(str(out_dir / f"{frame_idx}.txt"), nearest(frame_idx), fmt="%.10f")

print(f"\nWrote {len(all_indices)} pose files -> {out_dir}/")
first = np.loadtxt(str(out_dir / f"{gsa_indices[0]}.txt"))
print(f"Pose for frame {gsa_indices[0]}:")
print(first.round(4))
print("\nDone.")
