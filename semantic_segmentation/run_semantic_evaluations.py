import os
import shutil
import itertools
import subprocess
from datetime import datetime

from common import DATASETS_2D, DATASETS_3D


DATASETS = [*DATASETS_2D, *DATASETS_3D]


def write_batch_script(dataset, out_path, checkpoint, experiment_folder, use_lora, dry):
    "Writing scripts with different medico-sam semantic evaluations."
    batch_script = f"""#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH --constraint=80gb
#SBATCH --job-name=semsam_{dataset}

source ~/.bashrc
micromamba activate sam \n"""

    # python script
    python_script = "python ../evaluation/" + \
        ("semantic_segmentation_2d.py " if dataset in DATASETS_2D else "semantic_segmentation_3d.py ")
    python_script += f"-e {experiment_folder} "  # experiment folder
    python_script += "-m vit_b "  # name of the model configuration
    python_script += f"-c {checkpoint} "  # add finetuned checkpoints for semantic segmentation
    python_script += f"-d {dataset} "   # add name of the dataset
    if use_lora:  # whether to use lora for finetuning for semantic segmentation
        python_script += "--lora_rank 16 "

    batch_script += python_script   # let's add the python script to the bash script

    _op = out_path[:-3] + f"_{dataset}_lora-{use_lora}.sh"
    with open(_op, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", _op]
    if not dry:
        subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)
    script_name = "medico-sam-evaluation"
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")
    return batch_script


def submit_slurm(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    if args.dataset is not None:
        assert args.dataset in DATASETS
        datasets = [args.dataset]
    else:
        datasets = DATASETS

    lora_choices = [True, False]
    model_types = ["sam", "medico-sam-8g", "medico-sam-1g", "simplesam", "medsam"]

    for (per_dataset, model_type, use_lora) in itertools.product(datasets, model_types, lora_choices):
        mchoice = "vit_b"
        if per_dataset in DATASETS_3D:
            mchoice = "vit_b_3d_lora_16" if use_lora else "vit_b_3d_all"

        base_dir = os.path.join(args.save_root, "lora_finetuning" if use_lora else "full_finetuning", model_type)
        checkpoint = os.path.join(base_dir, "checkpoints", mchoice, f"{per_dataset}_semanticsam", "best.pt")
        assert os.path.exists(checkpoint), checkpoint

        print(f"Running for {per_dataset}: {model_type}")
        write_batch_script(
            dataset=per_dataset,
            out_path=get_batch_script_names(tmp_folder),
            checkpoint=checkpoint,
            experiment_folder=os.path.join(base_dir, "inference", per_dataset),
            use_lora=use_lora,
            dry=args.dry,
        )


def main(args):
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_slurm(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-s", "--save_root", type=str, default="/mnt/vast-nhr/projects/cidas/cca/models/semantic_sam")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
