#!/bin/bash

#SBATCH --job-name=entropy      # Job name
#SBATCH --output=output.txt     # Output file
#SBATCH --error=error.txt       # Error file
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --time=12:00:00         # Time limit hrs:min:sec
#SBATCH --mem=4G                # Memory limit
#SBATCH --gres=gpu:a100:1       # Request 1 GPU
#SBATCH -p general              # Specify the partition

# Load conda if available
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <your_env_name>

# (Optional) Set PYTHONPATH if needed
# export PYTHONPATH=$PYTHONPATH:/path/to/your/module

# Run script with arguments
python mlp.py --D="[3, 6, 9]" \
        --num_samples="[8192]" \
        --batch_size="[32]" \
        --num_steps="[10]" \
        --seed="[1,7,17,23]" \
        --epochs_list="[5, 10, 20, 30, 40, 80, 120, 160, 200]" \
        --maxL_prefactor="[False]" \
        --dist_scale="[0.0]" \
        --save_dir ../files/lab/GM_exp

# Also works for cifar10.py
python mnist.py \
        --batch_size="[32]" \
        --epochs_list="[5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]" \
        --num_steps="[1]" \
        --maxL_prefactor="[False]" \
        --seed="[1,7,17,23]" \
        --num_samples="[10]" \
        --save_dir ../files/lab/MNIST_exp