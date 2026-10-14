#!/bin/bash
#SBATCH --job-name=vggt_depth_eval
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodelist=node1
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

module load anaconda3-2024.2
module load cuda-12.8

export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/mazaveri/hpc-prog/sankarshan/repos/vggt:$PYTHONPATH

PYBIN=/home/mazaveri/.conda/envs/vlm/bin/python

echo "=== DIAGNOSTICS ==="
echo "PYBIN: $PYBIN"
$PYBIN -c "
import sys
print('sys.executable:', sys.executable)
try:
    import einops, torch
    print('einops found at:', einops.__file__)
    print('torch found at:', torch.__file__)
except ImportError as e:
    print('IMPORT FAILED:', e)
"
echo "=== END DIAGNOSTICS ==="

cd /home/mazaveri/hpc-prog/sankarshan
mkdir -p outputs/eval

$PYBIN -u eval_vggt_depth.py