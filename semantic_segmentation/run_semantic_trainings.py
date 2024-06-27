import os
import shutil
import itertools
import subprocess
from datetime import datetime


DATASETS = ["oimhs", "isic", "dca1", "cbis_ddsm", "m2caiseg", "btcv", "osic_pulmofib"]


def write_batch_script(out_path, _name, save_root, checkpoint, ckpt_name):
    "Writing scripts with different medico-sam finetunings."
    batch_script = f"""#!/bin/bash
#SBATCH -t 14-00:00:00
#SBATCH --mem 64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH -c 16
#SBATCH --qos=14d
#SBATCH --constraint=80gb
#SBATCH --job-name={os.path.split(_name)[-1]}

source activate sam \n"""

    # python script
    python_script = f"python {_name}.py "

    # save root folder
    python_script += f"-s {os.path.join(save_root, ckpt_name)} "

    # name of the model configuration
    python_script += "-m vit_b "

    # add pretrained checkpoints
    if checkpoint is not None:
        python_script += f"-c {checkpoint} "

    # let's add the python script to the bash script
    batch_script += python_script

    _op = out_path[:-3] + f"_{os.path.split(_name)[-1]}.sh"
    with open(_op, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", _op]
    subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "medico-sam-finetuning"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    script_combinations = {
        # 2d datasets
        "oimhs": "train_oimhs",
        "isic": "train_isic",
        "dca1": "train_dca1",
        "cbis_ddsm": "train_cbis_ddsm",
        "m2caiseg": "train_m2caiseg",
        # 3d datasets
        "btcv": "train_btcv",
        "osic_pulmofib": "train_osic_pulmofib",
    }
    if args.dataset is not None:
        assert args.dataset in DATASETS
        datasets = [args.dataset]
    else:
        datasets = DATASETS

    if args.checkpoint is not None:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = {
            "sam": None,
            "medico-sam": "medico-sam/multi_gpu/checkpoints/vit_b/medical_generalist_sam_multi_gpu/best_exported.pt",
            "simplesam": "simplesam/multi_gpu/checkpoints/vit_b/medical_generalist_simplesam_multi_gpu/best_exported.pt",  # noqa
            "medsam": "medsam/multi_gpu/checkpoints/vit_b/medical_generalist_medsam_multi_gpu/best_exported.pt",
        }

    for (per_dataset, ckpt_name) in itertools.product(datasets, checkpoints.keys()):
        script_name = script_combinations[per_dataset]
        checkpoint = None if checkpoints[ckpt_name] is None else os.path.join(args.save_root, checkpoints[ckpt_name])

        print(f"Running for {script_name}")
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            _name=script_name,
            save_root=os.path.join(args.save_root, "semantic_sam"),
            checkpoint=checkpoint,
            ckpt_name=ckpt_name,
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
    parser.add_argument("-s", "--save_root", type=str, default="/scratch/share/cidas/cca/models")
    args = parser.parse_args()
    main(args)
