#!/bin/bash
#SBATCH --job-name=pipeline_4scenes
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --gres=shard:50
#SBATCH --nodelist=node1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED

REPLICA_ROOT=/home/mazaveri/hpc-prog/sankarshan/data/replica/Replica_OCC/sequences
REPLICA_CONFIG_VGGT=/home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica_occ_vggt.yaml
GSA_PATH=/home/mazaveri/hpc-prog/sankarshan/repos/Grounded-Segment-Anything
CG_DIR=/home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs/conceptgraph
VLM_PY=/home/mazaveri/.conda/envs/vlm/bin/python
CG_PY=/home/mazaveri/.conda/envs/conceptgraph/bin/python

for SCENE in room1 room2 office0 office1; do
    echo "========================================"
    echo "Starting scene: $SCENE"
    echo "========================================"

    # Stage 1: Export VGGT-predicted poses
    echo "[$SCENE] Stage 1: Exporting VGGT poses..."
    $VLM_PY /home/mazaveri/hpc-prog/sankarshan/scripts/export_vggt_poses.py \
        --scene $SCENE
    echo "[$SCENE] Stage 1 done. Pose count:"
    ls ${REPLICA_ROOT}/${SCENE}/pose_vggt/ | wc -l

    # Stage 2: GSA detection
    echo "[$SCENE] Stage 2: Running GSA..."
    cd $CG_DIR
    $CG_PY scripts/generate_gsa_results.py \
        --dataset_root   $REPLICA_ROOT \
        --dataset_config $REPLICA_CONFIG_VGGT \
        --scene_id       $SCENE \
        --class_set      none \
        --stride         5
    echo "[$SCENE] Stage 2 done."

    # Stage 3: cfslam fusion
    echo "[$SCENE] Stage 3: Running cfslam..."
    $CG_PY slam/cfslam_pipeline_batch.py \
        dataset_root=$REPLICA_ROOT \
        dataset_config=$REPLICA_CONFIG_VGGT \
        scene_id=$SCENE \
        spatial_sim_type=overlap \
        mask_conf_threshold=0.95 \
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
        stride=20
    echo "[$SCENE] Stage 3 done."
    echo "[$SCENE] Output files:"
    ls -lh ${REPLICA_ROOT}/${SCENE}/pcd_saves/

done

echo "All 4 scenes complete."
