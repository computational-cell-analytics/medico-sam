#!/bin/bash
#SBATCH --job-name=medsam
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
python train_medical_medsam_multi_gpu.py -s /scratch/share/cidas/cca/models/medsam/multi_gpu
