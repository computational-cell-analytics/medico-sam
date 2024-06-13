import re
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from datetime import datetime


ALL_SCRIPTS = ["precompute_embeddings", "iterative_prompting"]

ROOT = "/scratch/share/cidas/cca"


def write_batch_script(
    out_path, inference_setup, checkpoint, model_type, experiment_folder, dataset_name, use_masks=False
):
    "Writing scripts with different fold-trainings for medico-sam evaluation"
    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH --constraint=80gb
#SBATCH --job-name={inference_setup}

source ~/.bashrc
mamba activate sam \n"""

    # python script
    inference_script_path = os.path.join(Path(__file__).parent, f"{inference_setup}.py")
    python_script = f"python {inference_script_path} "

    _op = out_path[:-3] + f"_{inference_setup}.sh"

    # add the finetuned checkpoint
    python_script += f"-c {checkpoint} "

    # name of the model configuration
    python_script += f"-m {model_type} "

    # experiment folder
    python_script += f"-e {experiment_folder} "

    # IMPORTANT: choice of the dataset
    python_script += f"-d {dataset_name} "

    # use logits for iterative prompting
    if inference_setup == "iterative_prompting" and use_masks:
        python_script += "--use_masks "

    # let's add the python script to the bash script
    batch_script += python_script

    with open(_op, "w") as f:
        f.write(batch_script)

    # we run the first prompt for iterative once starting with point, and then starting with box (below)
    if inference_setup == "iterative_prompting":
        batch_script += "--box "

        new_path = out_path[:-3] + f"_{inference_setup}_box.sh"
        with open(new_path, "w") as f:
            f.write(batch_script)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "medico-sam-inference"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def get_checkpoint_path(experiment_set, model_type, n_gpus):
    # let's set the experiment type - either using the generalist (or specific models) or using vanilla model
    if experiment_set == "generalist":
        if n_gpus == 1:
            checkpoint = os.path.join(
                ROOT, "models/medico-sam/single_gpu/checkpoints",
                model_type, "medical_generalist_sam_single_gpu/best.pt"
            )
        elif n_gpus == 8:
            checkpoint = os.path.join(
                ROOT, "models/medico-sam/multi_gpu/checkpoints",
                model_type, "medical_generalist_sam_multi_gpu/best.pt"
            )
        else:  # just a test model
            checkpoint = os.path.join(
                ROOT, "models", "test", "checkpoints", model_type, "medical_generalist_sam", "best.pt"
            )

    elif experiment_set == "vanilla":
        checkpoint = None

    elif experiment_set == "medsam":
        checkpoint = "/scratch-grete/projects/nim00007/sam/models/medsam/medsam_vit_b.pth"

    else:
        raise ValueError("Choose from 'generalist' / 'vanilla' / 'medsam'.")

    if checkpoint is not None:
        assert os.path.exists(checkpoint), checkpoint

    return checkpoint


def submit_slurm(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    # parameters to run the inference scripts
    dataset_name = args.dataset_name  # name of the dataset in lower-case
    model_type = args.model_type
    experiment_set = args.experiment_set  # infer using generalist or vanilla models
    n_gpus = args.gpus

    if args.checkpoint_path is None:
        checkpoint = get_checkpoint_path(experiment_set, n_gpus)
    else:
        checkpoint = args.checkpoint_path

    if args.experiment_path is None:
        experiment_folder = os.path.join(ROOT, "experiments", "v1", experiment_set, dataset_name, model_type)
    else:
        experiment_folder = args.experiment_path

    for current_setup in ALL_SCRIPTS:
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            inference_setup=current_setup,
            checkpoint=checkpoint,
            model_type=model_type,
            experiment_folder=experiment_folder,
            dataset_name=dataset_name,
            use_masks=args.use_masks
            )

    # the logic below automates the process of first running the precomputation of embeddings, and only then inference.
    job_id = []
    for i, my_script in enumerate(sorted(glob(tmp_folder + "/*"))):
        cmd = ["sbatch", my_script]

        if i > 0:
            cmd.insert(1, f"--dependency=afterany:{job_id[0]}")

        cmd_out = subprocess.run(cmd, capture_output=True, text=True)
        print(cmd_out.stdout if len(cmd_out.stdout) > 1 else cmd_out.stderr)

        if i == 0:
            job_id.append(re.findall(r'\d+', cmd_out.stdout)[0])


def main(args):
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_slurm(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, required=True)
    parser.add_argument("-e", "--experiment_set", type=str, required=True)
    parser.add_argument("--use_masks", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--experiment_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
