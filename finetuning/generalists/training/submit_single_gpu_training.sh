#!/bin/bash
#SBATCH --job-name=medico-sam
#SBATCH -t 14-00:00:00
#SBATCH --nodes=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH -c 64
#SBATCH --mem 128G
#SBATCH --qos=14d
#SBATCH --constraint=80gb

source ~/.bashrc
mamba activate sam
python train_medical_generalist.py -s /scratch/share/cidas/cca/models/medico-sam/single_gpu
