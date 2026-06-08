#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --partition=H100,A100,L40S

source /home/ids/gmargari-24/airway_project/3env/bin/activate

echo "--- environment diagnostic ---"
echo "which python: $(which python)"
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
python -c "import torch; print('torch:', torch.__file__, torch.__version__)"
python -c "
import torch
try:
    import torch._dynamo
    print('dynamo import: ok')
except Exception as e:
    print('dynamo import: FAILED', type(e).__name__, e)
"
echo "--- end diagnostic ---"