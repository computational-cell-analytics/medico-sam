import os
import shutil
import itertools
import subprocess
from datetime import datetime

from common import DATASETS_2D, DATASETS_3D, MODELS_ROOT


def write_batch_script(
    out_path, dataset_name, save_root, checkpoint, ckpt_name, use_lora, dry, iterations, uno,
):
    "Writing scripts with different medico-sam finetunings."
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
#SBATCH --job-name=semsam_{dataset_name}

source ~/.bashrc
micromamba activate super \n"""

    # python script
    python_script = "python train_semantic_segmentation.py "
    python_script += f"-d {dataset_name} "  # name of the dataset
    python_script += f"-s {os.path.join(save_root, ckpt_name)} "  # save root folder
    python_script += "-m vit_b "  # name of the model configuration
    if checkpoint is not None:  # add pretrained checkpoints
        python_script += f"-c {checkpoint} "

    if uno:  # whether to train with one image only.
        python_script += "--uno "

    if iterations is not None:  # number of iterations to train the model for
        python_script += f"--iterations {iterations} "

    if use_lora:  # whether to use lora for finetuning for semantic segmentation
        python_script += "--lora_rank 16 "

    batch_script += python_script  # let's add the python script to the bash script

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
        checkpoints = {"custom experiment": args.checkpoint}
    else:
        checkpoints = {
            # default SAM model
            "sam": None,
            # our finetuned models
            # NOTE: Next one model is the old medico-sam model.
            # "medico-sam-8g": "medico-sam/multi_gpu/checkpoints/vit_b/medical_generalist_sam_multi_gpu/best_exported.pt",  # noqa
            # NOTE: Next one model is the new medico-sam (all data) generalist model.
            "medico-samv2-full": "medico-sam/v2/multi_gpu/checkpoints/vit_b/medical_generalist_sam_multi_gpu/model.pt",
        }
        if not args.uno:  # i.e. we train 1 image models with the best chosen models only.
            # MedSAM's original model.
            checkpoints["medsam"] = "medsam/original/medsam_vit_b.pth"
            # our finetuned models.
            # checkpoints["medico-sam-1g"] = "medico-sam/single_gpu/checkpoints/vit_b/medical_generalist_sam_single_gpu/best.pt"  # noqa
            checkpoints["simplesam"] = "simplesam/multi_gpu/checkpoints/vit_b/medical_generalist_simplesam_multi_gpu/best_exported.pt"  # noqa

    lora_choices = [True, False]
    for (per_dataset, ckpt_name, use_lora) in itertools.product(datasets, checkpoints.keys(), lora_choices):
        checkpoint = None if checkpoints[ckpt_name] is None else os.path.join(MODELS_ROOT, checkpoints[ckpt_name])

        print(f"Running for experiment name '{ckpt_name}'")
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            dataset_name=per_dataset,
            save_root=os.path.join(
                args.save_root,
                "semantic_sam" + ("_uno" if args.uno else ""),
                "v2",  # NOTE: v2 models are the UNETR style models.
                "lora_finetuning" if use_lora else "full_finetuning"
            ),
            checkpoint=checkpoint,
            ckpt_name=ckpt_name,
            use_lora=use_lora,
            dry=args.dry,
            iterations=args.iterations,
            uno=args.uno,
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
    parser.add_argument("--iterations", type=int, default=int(5e4))
    parser.add_argument("--uno", action="store_true")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
