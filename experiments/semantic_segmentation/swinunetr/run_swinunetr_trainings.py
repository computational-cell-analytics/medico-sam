import os
import sys
import shutil
from datetime import datetime

import subprocess


sys.path.append("..")


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)
    script_name = "medico-sam-finetuning"
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")
    return batch_script


def submit_slurm(tmp_folder, dry):
    """Submit python scripts to cluster via SLURM.
    """
    from common import DATASETS_2D, DATASETS_3D, MODELS_ROOT

    datasets = [*DATASETS_3D, *DATASETS_2D]
    save_root = os.path.join(MODELS_ROOT, "swinunetr")

    for dataset in datasets:
        # Now, get the template batch script.
        batch_script = f"""#!/bin/bash
#SBATCH -t 4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH --constraint=80gb
#SBATCH --qos=96h
#SBATCH --job-name=swinunetr_{dataset}

source ~/.bashrc
micromamba activate super \n"""

        # Prepare the python script.
        python_script = "python train_swinunetr.py "
        python_script += f"-d {dataset} "
        python_script += f"-s {save_root} "

        # Add it to the batch script template.
        batch_script += python_script

        _op = get_batch_script_names(tmp_folder)[:-3] + f"_{dataset}.sh"
        with open(_op, "w") as f:
            f.write(batch_script)

        cmd = ["sbatch", _op]
        if not dry:
            subprocess.run(cmd)


def main():
    tmp_folder = "./gpu_jobs"
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    submit_slurm(tmp_folder, dry=False)


if __name__ == "__main__":
    main()
