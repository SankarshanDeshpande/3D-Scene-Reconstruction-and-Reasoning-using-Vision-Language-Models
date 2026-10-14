#!/bin/bash
#SBATCH --job-name=regen_pc
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

module load anaconda3-2024.2
module load cuda-12.8

export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/mazaveri/hpc-prog/sankarshan/repos/vggt:$PYTHONPATH

PYBIN=/home/mazaveri/.conda/envs/vlm/bin/python

echo "PYBIN: $PYBIN"
echo "NODE: $(hostname)"
$PYBIN -c "import torch; print('torch:', torch.__file__); print('cuda available:', torch.cuda.is_available())"

cd /home/mazaveri/hpc-prog/sankarshan

$PYBIN -u regen_pointcloud.py
