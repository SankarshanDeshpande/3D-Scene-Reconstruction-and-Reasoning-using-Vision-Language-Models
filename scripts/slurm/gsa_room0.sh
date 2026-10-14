#!/bin/bash
#SBATCH --job-name=gsa_room0
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

source /home/mazaveri/.conda/envs/conceptgraph/bin/activate conceptgraph 2>/dev/null || conda activate conceptgraph

export REPLICA_ROOT=/home/mazaveri/hpc-prog/sankarshan/data/replica/Replica_OCC/sequences
export REPLICA_CONFIG_PATH=/home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica_occ.yaml
export GSA_PATH=/home/mazaveri/hpc-prog/sankarshan/repos/Grounded-Segment-Anything

cd /home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs/conceptgraph
python scripts/generate_gsa_results.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id room0 \
    --class_set none \
    --stride 5
