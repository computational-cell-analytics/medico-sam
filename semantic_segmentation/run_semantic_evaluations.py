import os
import shutil
import itertools
import subprocess
from datetime import datetime

from common import DATASETS, MODELS_ROOT


def write_batch_script(dataset, out_path, checkpoint, experiment_folder, use_lora, dry):
    "Writing scripts with different medico-sam semantic evaluations."
    batch_script = f"""#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --mem 64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH -c 16
#SBATCH --constraint=80gb
#SBATCH --job-name={dataset}_semsam

source ~/.bashrc
micromamba activate sam \n"""

    # python script
    python_script = "python ../evaluation/semantic_segmentation_2d.py "

    # experiment folder
    python_script += f"-e {experiment_folder} "

    # name of the model configuration
    python_script += "-m vit_b "

    # add finetuned checkpoints for semantic segmentation
    python_script += f"-c {checkpoint} "

    # add name of the dataset
    python_script += f"-d {dataset} "

    # whether to use lora for finetuning for semantic segmentation
    if use_lora:
        python_script += "--lora_rank 4 "

    # let's add the python script to the bash script
    batch_script += python_script

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
        base_dir = os.path.join(args.save_root, "lora_finetuning" if use_lora else "full_finetuning", model_type)
        checkpoint = os.path.join(base_dir, "checkpoints", "vit_b", f"{per_dataset}_semanticsam", "best.pt")
        assert os.path.exists(checkpoint), checkpoint

        print(f"Running for {per_dataset}")
        write_batch_script(
            dataset=per_dataset,
            out_path=get_batch_script_names(tmp_folder),
            checkpoint=checkpoint,
            experiment_folder=base_dir,
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
    parser.add_argument("-s", "--save_root", type=str, default="/scratch/share/cidas/cca/models/semantic_sam")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
