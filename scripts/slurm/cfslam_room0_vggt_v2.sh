#!/bin/bash
#SBATCH --job-name=cfslam_r0_vggt_v2
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --gres=shard:50
#SBATCH --nodelist=node2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED

cd /home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs/conceptgraph

/home/mazaveri/.conda/envs/conceptgraph/bin/python slam/cfslam_pipeline_batch.py \
    dataset_root=/home/mazaveri/hpc-prog/sankarshan/data/replica/Replica_OCC/sequences \
    dataset_config=/home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica_occ_vggt.yaml \
    scene_id=room0 \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.25 \
    match_method=sim_sum \
    sim_threshold=1.2 \
    dbscan_eps=0.2 \
    gsa_variant=none \
    skip_bg=true \
    max_bbox_area_ratio=0.5 \
    merge_interval=20 \
    merge_overlap_thresh=0.1 \
    merge_visual_sim_thresh=0.3 \
    merge_text_sim_thresh=0.3 \
    stride=5
