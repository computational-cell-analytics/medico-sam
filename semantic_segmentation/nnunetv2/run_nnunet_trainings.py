import os
import sys
import shutil
import subprocess
from datetime import datetime


sys.path.append("..")


def write_batch_script(out_path, dataset_name, fold_choice, mode, dry):
    "Writing scripts with different nnUNet training and inference runs."
    batch_script = f"""#!/bin/bash
#SBATCH -t 4-00:00:00
#SBATCH --mem 128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH -c 32
#SBATCH --constraint=80gb
#SBATCH --qos=96h
#SBATCH -x ggpu[150,152,157,212]
#SBATCH --job-name=nnunet_{dataset_name}

source ~/.bashrc
micromamba activate nnunet \n"""

    # python script
    python_script = "python train_nnunetv2.py "
    python_script += f"-d {dataset_name} "  # name of the dataset.
    python_script += f"--fold {fold_choice} "  # the choice of nnunet fold to train.
    if mode is not None and isinstance(mode, list):
        python_script += " ".join(mode) + " "  # choose either '--preprocess' / '--train' / '--predict' (or multiple).

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
    script_name = "nnunet-training"
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")
    return batch_script


def submit_slurm(args, unknown_args, tmp_folder):
    "Submit python script that needs gpus with given inputs on a slurm node."
    from common import DATASETS_2D, DATASETS_3D
    all_datasets = [*DATASETS_3D, *DATASETS_2D]

    if args.dataset is not None:
        assert args.dataset in all_datasets
        datasets = [args.dataset]
    else:
        datasets = all_datasets

    mode = None
    if len(unknown_args) > 0:
        for arg in unknown_args:
            assert arg in ["--preprocess", "--train", "--preprocess"], f"'{arg}' is not a supported argument."
        mode = unknown_args
    else:
        raise ValueError(
            "You need to parse arguments to work with nnUNet (eg. '--preprocess / '--train' / '--predict'.)"
        )

    for per_dataset in datasets:
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            dataset_name=per_dataset,
            dry=args.dry,
            fold_choice=args.fold,
            mode=mode,
        )


def main(args, unknown_args):
    tmp_folder = "./gpu_jobs"
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    submit_slurm(args, unknown_args, tmp_folder)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--dry", action="store_true")
    args, unknown_args = parser.parse_known_args()  # we catch nnunet-related actions under 'unknown_args'.
    main(args, unknown_args)
