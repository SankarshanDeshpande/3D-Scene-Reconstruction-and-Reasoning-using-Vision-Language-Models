#!/bin/bash
#SBATCH --job-name=build_scenegraph_room0
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodelist=node1
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00

PYTHON_BIN=/home/mazaveri/.conda/envs/conceptgraph/bin/python
VLM_PYTHON=/home/mazaveri/.conda/envs/vlm/bin/python
OLLAMA_BIN=/home/mazaveri/.conda/envs/vlm/bin/ollama

echo "PYTHON: $PYTHON_BIN"
echo "NODE: $(hostname)"

# Start Ollama inside this same job/node context
echo "Starting Ollama server..."
$OLLAMA_BIN serve > /home/mazaveri/hpc-prog/sankarshan/ollama_server_job.log 2>&1 &
OLLAMA_PID=$!
sleep 8

# Verify it's reachable before proceeding
$PYTHON_BIN -c "
import urllib.request, time
for attempt in range(5):
    try:
        r = urllib.request.urlopen('http://localhost:11434', timeout=5)
        print('Ollama reachable:', r.read().decode())
        break
    except Exception as e:
        print(f'Attempt {attempt+1} failed:', e)
        time.sleep(3)
else:
    print('FATAL: Ollama never became reachable')
"

cd /home/mazaveri/hpc-prog/sankarshan/repos/concept-graphs/conceptgraph
$PYTHON_BIN /home/mazaveri/hpc-prog/sankarshan/run_build_scenegraph.py

# Clean up the Ollama process this job started
kill $OLLAMA_PID 2>/dev/null
