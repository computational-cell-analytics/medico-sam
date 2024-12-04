import os
import sys
import shutil
import itertools
import subprocess
from datetime import datetime

from micro_sam2.util import CFG_PATHS


sys.path.append("..")


PROMPT_CHOICES = ["box", "point"]


def _get_slurm_template(job_name, env_name):
    batch_script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 4-00:00:00
#SBATCH -c 32
#SBATCH --mem 128G
#SBATCH -p grete:shared
#SBATCH --constraint=80gb
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH --qos=96h
#SBATCH -x ggpu[150,201,202,212]
#SBATCH --job-name={job_name}

source ~/.bashrc
micromamba activate {env_name} \n"""

    return batch_script


def write_batch_script(
    out_path,
    dataset_name,
    is_2d,
    model_type,
    backbone,
    experiment_folder,
    prompt_choice,
    dry_run,
    time_delay,
    use_mask,
):
    """Writing batch submission scripts for SAM2 evaluations.
    """
    batch_script = _get_slurm_template(job_name="sam2_mi_evaluation", env_name="sam2")

    # add delay by a few seconds (to avoid io blocking in container data access)
    batch_script += f"sleep {time_delay}s \n"

    # python script to run.
    pscript = "python " + ("evaluate_interactive_2d.py " if is_2d else "evaluate_interactive_3d.py ")
    pscript += f"-d {dataset_name} "  # the name of dataset.
    pscript += f"-m {model_type} "  # the choice of model.
    pscript += f"-b {backbone} "  # the SAM2 backbone.
    pscript += f"-e {experiment_folder} "  # the path to directory where results will be cached.

    if is_2d:
        pscript += f"-p {prompt_choice} "  # choice of prompts for interactive segmentation.
    else:
        pscript += f"-p {prompt_choice} "  # the starting prompt choice

    if use_mask:  # Whether to use masks for iterative prompting.
        pscript += "--use_masks "

    batch_script += pscript

    opath = out_path[:-3] + f"_sam2_{dataset_name}_{model_type}_{backbone}_{prompt_choice}.sh"
    with open(opath, "w") as f:
        f.write(batch_script)

    if not dry_run:
        cmd = ["sbatch", opath]
        subprocess.run(cmd)


def get_batch_script_names(out_folder, script_name="sam2_evaluation"):
    out_folder = os.path.expanduser(out_folder)
    os.makedirs(out_folder, exist_ok=True)
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(out_folder, f"{tmp_name}.sh")
    return batch_script


def submit_to_slurm(
    dataset_name, backbone, experiment_folder, prompt_choice, dry_run,
):
    """Submit python scripts with given arguments on a compute node via slurm.
    """
    from util import VALID_DATASETS

    out_folder = "./gpu_jobs"
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)

    # Create parameter combinations if the user does not request for specific combinations.
    dnames = VALID_DATASETS if dataset_name is None else dataset_name
    backbones = list(CFG_PATHS.keys()) if backbone is None else backbone
    run_choices = PROMPT_CHOICES if prompt_choice is None else prompt_choice
    use_masks = [True, False]

    time_delay = 0
    for dname, bb, rchoice, use_mask in itertools.product(dnames, backbones, run_choices, use_masks):
        mtype = "hvit_b"  # NOTE: for the current experiments, we stick to 'hvit_b' model.

        write_batch_script(
            out_path=get_batch_script_names(out_folder),
            dataset_name=dname,
            is_2d=True,  # TODO
            model_type=mtype,
            backbone=bb,
            experiment_folder=os.path.join(experiment_folder, bb, mtype, dname),
            prompt_choice=rchoice,
            dry_run=dry_run,
            time_delay=time_delay,
            use_mask=use_mask,
        )
        # time_delay += 5  # add time delay by 'n' seconds per job run.


def main(args):
    submit_to_slurm(
        dataset_name=args.dataset_name,
        backbone=args.backbone,
        experiment_folder=args.experiment_folder,
        prompt_choice=args.prompt_choice,
        dry_run=args.dry,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, nargs="*", default=None)
    parser.add_argument("-b", "--backbone", type=str, nargs="*", default=None)
    parser.add_argument(
        "-e", "--experiment_folder", type=str, default="/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam/sam2"
    )
    parser.add_argument("-p", "--prompt_choice", type=str, nargs="*", default=None)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
