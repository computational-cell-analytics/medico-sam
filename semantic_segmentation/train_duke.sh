#!/bin/bash
#SBATCH --partition=grete:shared
#SBATCH -G A100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=nim00007
#SBATCH --constraint=80gb
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem 128G
#SBATCH --job-name=train-duke-liver


source /home/nimlufre/.bashrc
conda activate sam

python /home/nimlufre/medico-sam/semantic_segmentation/train_duke_liver.py \
  --use_lora \
  --save_root "/scratch-grete/usr/nimlufre/medico_sam/semantic_segmentation/"