import re
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from datetime import datetime


ALL_SCRIPTS = ["iterative_prompting"]
ROOT = "/mnt/vast-nhr/projects/cidas/cca"


def write_batch_script(
    out_path,
    inference_setup,
    checkpoint,
    model_type,
    experiment_folder,
    dataset_name,
    use_masks=False,
    use_sam_med2d=False,
    adapter=False,
):
    "Writing scripts with different fold-trainings for medico-sam evaluation"
    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH --job-name={inference_setup}

source ~/.bashrc
micromamba activate sam \n"""

    # python script
    inference_script_path = os.path.join(Path(__file__).parent, f"{inference_setup}.py")
    python_script = f"python {inference_script_path} "
    python_script += f"-m {model_type} "  # name of the model configuration
    python_script += f"-e {experiment_folder} "  # experiment folder
    python_script += f"-d {dataset_name} "  # choice of the dataset
    python_script += f"-c {checkpoint} "  # add the finetuned checkpoint

    if inference_setup == "iterative_prompting" and use_masks:  # use logits for iterative prompting
        python_script += "--use_masks "

    if use_sam_med2d:  # use SAM-Med2d for inference
        python_script += "--use_sam_med2d "
        if adapter:
            python_script += "--adapter "

    batch_script += python_script  # let's add the python script to the bash script

    _op = out_path[:-3] + f"_{inference_setup}.sh"
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


def get_checkpoint_path_and_params(experiment_set, model_type, n_gpus):
    extra_params = {}

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
                model_type, "medical_generalist_sam_multi_gpu/best_exported.pt"
            )
        else:
            raise ValueError(n_gpus)

    elif experiment_set == "simplesam":
        if n_gpus == 1:
            checkpoint = os.path.join(
                ROOT, "models/simplesam/single_gpu/checkpoints",
                model_type, "medical_generalist_simplesam_single_gpu/best.pt"
            )
        elif n_gpus == 8:
            checkpoint = os.path.join(
                ROOT, "models/simplesam/multi_gpu/checkpoints",
                model_type, "medical_generalist_simplesam_multi_gpu/best_exported.pt"
            )
        else:
            raise ValueError(n_gpus)

    elif experiment_set == "medsam-self":
        if n_gpus == 1:
            checkpoint = os.path.join(
                ROOT, "models/medsam/single_gpu/checkpoints",
                model_type, "medical_generalist_medsam_single_gpu/best.pt"
            )
        elif n_gpus == 8:
            checkpoint = os.path.join(
                ROOT, "models/medsam/multi_gpu/checkpoints",
                model_type, "medical_generalist_medsam_multi_gpu/best_exported.pt"
            )
        else:
            raise ValueError(n_gpus)

    elif experiment_set == "vanilla":
        checkpoint = None

    elif experiment_set == "medsam":
        checkpoint = "/scratch-grete/projects/nim00007/sam/models/medsam/medsam_vit_b.pth"

    elif experiment_set == "sam-med2d":
        checkpoint = "/scratch-grete/projects/nim00007/sam/models/sam-med2d/ft-sam_b.pth"
        extra_params["use_sam_med2d"] = True

    elif experiment_set == "sam-med2d-adapter":
        checkpoint = "/scratch-grete/projects/nim00007/sam/models/sam-med2d/sam-med2d_b.pth"
        extra_params["use_sam_med2d"] = True
        extra_params["adapter"] = True

    else:
        raise ValueError(experiment_set)

    if checkpoint is not None:
        assert os.path.exists(checkpoint), checkpoint

    return checkpoint, extra_params


def submit_slurm(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    _add_dependency = False

    tmp_folder = "./gpu_jobs"

    # parameters to run the inference scripts
    dataset_name = args.dataset_name  # name of the dataset in lower-case
    model_type = args.model_type
    experiment_set = args.experiment_set  # infer using generalist or vanilla models
    n_gpus = args.gpus
    if n_gpus is not None:
        n_gpus = int(n_gpus)

    if args.checkpoint_path is None:
        checkpoint, extra_params = get_checkpoint_path_and_params(
            experiment_set=experiment_set, model_type=model_type, n_gpus=n_gpus
        )
    else:
        checkpoint = args.checkpoint_path

    if args.experiment_path is None:
        if n_gpus is not None:
            ename = f"{experiment_set}_{n_gpus}"
        else:
            ename = experiment_set

        # NOTE: v1 was the first version of evaluation with all datasets having inference as expected
        # v2 are the additional experiments for iterative prompting with masks.
        experiment_folder = os.path.join(ROOT, "experiments", "v2", ename, dataset_name, model_type)
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
            use_masks=args.use_masks,
            **extra_params
            )

    if args.dry:
        return

    # the logic below automates the process of first running the precomputation of embeddings, and only then inference.
    job_id = []
    for i, my_script in enumerate(sorted(glob(tmp_folder + "/*"))):
        cmd = ["sbatch", my_script]

        if _add_dependency and i > 0:
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
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--experiment_path", type=str, default=None)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
