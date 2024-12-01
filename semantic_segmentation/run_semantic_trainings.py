import os
import shutil
import itertools
import subprocess
from datetime import datetime

from common import DATASETS_2D, DATASETS_3D, MODELS_ROOT


def write_batch_script(
    out_path, dataset_name, save_root, checkpoint, ckpt_name, use_lora, dry, iterations,
):
    "Writing scripts with different medico-sam finetunings."
    batch_script = f"""#!/bin/bash
#SBATCH -t 4-00:00:00
#SBATCH --mem 64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH -c 16
#SBATCH --constraint=80gb
#SBATCH --qos=96h
#SBATCH --job-name=semsam_{dataset_name}

source ~/.bashrc
micromamba activate sam \n"""

    # python script
    python_script = "python train_semantic_segmentation.py "

    # name of the dataset
    python_script += f"-d {dataset_name} "

    # save root folder
    python_script += f"-s {os.path.join(save_root, ckpt_name)} "

    # name of the model configuration
    python_script += "-m vit_b "

    # add pretrained checkpoints
    if checkpoint is not None:
        python_script += f"-c {checkpoint} "

    # number of iterations to train the model for
    if iterations is not None:
        python_script += f"--iterations {iterations} "

    # whether to use lora for finetuning for semantic segmentation
    if use_lora:
        python_script += "--lora_rank 16 "

    # let's add the python script to the bash script
    batch_script += python_script

    _op = out_path[:-3] + f"_{dataset_name}.sh"
    with open(_op, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", _op]

    if not dry:
        subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)
    script_name = "medico-sam-finetuning"
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")
    return batch_script


def submit_slurm(args, tmp_folder):
    "Submit python script that needs gpus with given inputs on a slurm node."
    all_datasets = [*DATASETS_3D, *DATASETS_2D]

    if args.dataset is not None:
        assert args.dataset in all_datasets
        datasets = [args.dataset]
    else:
        datasets = all_datasets

    if args.checkpoint is not None:
        checkpoints = {"experiment": args.checkpoint}
    else:
        checkpoints = {
            # default SAM model
            "sam": None,
            # our finetuned models
            "medico-sam-8g": "medico-sam/multi_gpu/checkpoints/vit_b/medical_generalist_sam_multi_gpu/best_exported.pt",
            "medico-sam-1g": "medico-sam/single_gpu/checkpoints/vit_b/medical_generalist_sam_single_gpu/best.pt",
            "simplesam": "simplesam/multi_gpu/checkpoints/vit_b/medical_generalist_simplesam_multi_gpu/best_exported.pt",  # noqa
            # MedSAM's original model
            "medsam": "medsam/original/medsam_vit_b.pth",
        }

    lora_choices = [True, False]
    for (per_dataset, ckpt_name, use_lora) in itertools.product(datasets, checkpoints.keys(), lora_choices):
        checkpoint = None if checkpoints[ckpt_name] is None else os.path.join(MODELS_ROOT, checkpoints[ckpt_name])

        print(f"Running for experiment name '{ckpt_name}'")
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            dataset_name=per_dataset,
            save_root=os.path.join(
                args.save_root, "semantic_sam", "lora_finetuning" if use_lora else "full_finetuning"
            ),
            checkpoint=checkpoint,
            ckpt_name=ckpt_name,
            use_lora=use_lora,
            dry=args.dry,
            iterations=args.iterations,
        )


def main(args):
    tmp_folder = "./gpu_jobs"
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    submit_slurm(args, tmp_folder)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-s", "--save_root", type=str, default="/mnt/vast-nhr/projects/cidas/cca/models")
    parser.add_argument("--iterations", type=int, default=int(1e4))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
