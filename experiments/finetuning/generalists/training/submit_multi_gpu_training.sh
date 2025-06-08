#!/bin/bash
#SBATCH --job-name=medico-sam
#SBATCH -t 14-00:00:00
#SBATCH --nodes=1
#SBATCH -p grete:shared
#SBATCH -G A100:8
#SBATCH -A gzz0001
#SBATCH --cpus-per-gpu 16
#SBATCH --mem-per-gpu 96G
#SBATCH --qos=14d
#SBATCH --constraint=80gb

source ~/.bashrc
mamba activate sam
python train_medical_multi_gpu_generalist.py -s /mnt/vast-nhr/projects/cidas/cca/models/medico-sam/v2/multi_gpu
