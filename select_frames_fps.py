"""
Compares naive even-spaced frame selection (what every run so far has used) against
a coverage-maximizing selection based on the GT camera poses ReplicaOcc already provides
for all 2000 frames. Pure CPU, no GPU/VGGT inference needed -- run this on the login node
to decide whether re-selecting frames is worth doing before spending compute on a rerun.

Idea: even-index spacing assumes uniform trajectory motion = uniform scene coverage, which
isn't guaranteed (the camera can dwell on one wall while barely glancing at another). Farthest
Point Sampling (FPS) over camera position + view direction picks the most spatially/angularly
DIVERSE 128 viewpoints from the full 2000, which should cover more of the room even at the
same frame budget.
"""

import os
import numpy as np

SCENES = ["room0", "room1", "room2", "office0", "office1"]
DATA_ROOT = "/home/mazaveri/hpc-prog/sankarshan/data/replica/Replica_OCC/sequences"
N_SAMPLES = 128


def load_pose(path):
    """Parse a pose/*.txt file -- 4x4 camera-to-world matrix, whitespace-separated."""
    vals = np.loadtxt(path).flatten()
    return vals[:16].reshape(4, 4)


def camera_position_and_view(pose):
    position = pose[:3, 3]
    view_dir = -pose[:3, 2]   # forward axis (convention-dependent sign; doesn't matter for diversity purposes)
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-9)
    return position, view_dir


def farthest_point_sampling(features, n_samples, seed=0):
    """Greedily pick n_samples rows from `features` maximizing min-distance to the
    already-selected set. Returns the selected row indices, in selection order."""
    n = features.shape[0]
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    dist = np.linalg.norm(features - features[selected[0]], axis=1)
    for _ in range(1, n_samples):
        next_idx = int(np.argmax(dist))
        selected.append(next_idx)
        new_dist = np.linalg.norm(features - features[next_idx], axis=1)
        dist = np.minimum(dist, new_dist)
    return selected


def even_spaced_indices(n_total, n_samples):
    return sorted({int(round(i * (n_total - 1) / (n_samples - 1))) for i in range(n_samples)})


def main():
    for scene in SCENES:
        pose_dir = os.path.join(DATA_ROOT, scene, "pose")
        pose_files = sorted(
            [f for f in os.listdir(pose_dir) if f.endswith(".txt")],
            key=lambda f: int(os.path.splitext(f)[0])
        )
        n_total = len(pose_files)

        positions = np.zeros((n_total, 3))
        view_dirs = np.zeros((n_total, 3))
        for i, f in enumerate(pose_files):
            pose = load_pose(os.path.join(pose_dir, f))
            positions[i], view_dirs[i] = camera_position_and_view(pose)

        # Normalize position scale roughly to room size, so position and view-direction
        # contribute comparably to the FPS distance metric.
        pos_scale = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)) + 1e-9
        features = np.concatenate([positions / pos_scale, view_dirs * 0.5], axis=1)

        even_idxs = even_spaced_indices(n_total, N_SAMPLES)
        fps_idxs = sorted(farthest_point_sampling(features, N_SAMPLES))

        # Compare coverage: for each selection, compute how far the FARTHEST unselected
        # frame's camera is from its nearest selected neighbor -- a lower value means
        # better coverage (no region of the trajectory is left too far from a sampled view).
        def worst_case_gap(idxs):
            sel_feat = features[idxs]
            all_dists = np.linalg.norm(features[:, None, :] - sel_feat[None, :, :], axis=2)
            nearest = all_dists.min(axis=1)
            return nearest.max(), nearest.mean()

        even_worst, even_mean = worst_case_gap(even_idxs)
        fps_worst, fps_mean = worst_case_gap(fps_idxs)

        print(f"[{scene}] even-spaced:  worst-case gap={even_worst:.4f}  mean gap={even_mean:.4f}")
        print(f"[{scene}] FPS-selected: worst-case gap={fps_worst:.4f}  mean gap={fps_mean:.4f}")
        print(f"[{scene}] improvement:  worst-case {100*(1-fps_worst/even_worst):.1f}% smaller, "
              f"mean {100*(1-fps_mean/even_mean):.1f}% smaller\n")

        # Save the FPS-selected global frame indices for use in regen_pointcloud.py
        out_path = f"/home/mazaveri/hpc-prog/sankarshan/outputs/eval/{scene}_fps_indices.txt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(",".join(str(i) for i in fps_idxs))
        print(f"[{scene}] saved FPS frame indices to {out_path}")


if __name__ == "__main__":
    main()
