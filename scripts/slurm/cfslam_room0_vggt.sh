#!/bin/bash
#SBATCH --job-name=cfslam_room0_vggt
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --gres=shard:50
#SBATCH --nodelist=node2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED

export REPLICA_ROOT=/home/mazaveri/hpc-prog/sankarshan/data/replica/Replica_OCC/sequences
export REPLICA_CONFIG_VGGT=/home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica_occ_vggt.yaml

cd /home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs/conceptgraph

/home/mazaveri/.conda/envs/conceptgraph/bin/python slam/cfslam_pipeline_batch.py \
    dataset_root=$REPLICA_ROOT \
    dataset_config=$REPLICA_CONFIG_VGGT \
    scene_id=room0 \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.95 \
    match_method=sim_sum \
    sim_threshold=1.2 \
    dbscan_eps=0.1 \
    gsa_variant=none \
    skip_bg=true \
    max_bbox_area_ratio=0.5 \
    merge_interval=20 \
    merge_visual_sim_thresh=0.8 \
    merge_text_sim_thresh=0.8 \
    stride=5
