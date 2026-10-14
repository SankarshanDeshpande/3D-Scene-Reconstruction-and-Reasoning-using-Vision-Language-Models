#!/bin/bash
#SBATCH --job-name=gsa_room0_vggt
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --gres=shard:60
#SBATCH --nodelist=node2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

export REPLICA_ROOT=/home/mazaveri/hpc-prog/sankarshan/data/replica/Replica_OCC/sequences
export REPLICA_CONFIG_VGGT=/home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica_occ_vggt.yaml
export GSA_PATH=/home/mazaveri/hpc-prog/sankarshan/repos/Grounded-Segment-Anything

cd /home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs/conceptgraph

/home/mazaveri/.conda/envs/conceptgraph/bin/python scripts/generate_gsa_results.py \
    --dataset_root   $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_VGGT \
    --scene_id       room0 \
    --class_set      none \
    --stride         5
