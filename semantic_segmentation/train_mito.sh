#!/bin/bash
#SBATCH --partition=grete-h100:shared
#SBATCH -G H100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem 128G
#SBATCH --job-name=train-mito


lr=1e-4

source /home/nimlufre/.bashrc
conda activate sam

python /home/nimlufre/medico-sam/semantic_segmentation/train_mito.py \
  --use_lora \
  --learning_rate $lr \
  --save_root "/scratch-grete/usr/nimlufre/medico_sam/mito_segmentation_lrs/$lr"