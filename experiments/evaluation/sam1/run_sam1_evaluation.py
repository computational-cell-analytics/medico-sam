import os
import sys
import shutil
import itertools
import subprocess


sys.path.append("../sam2")


PROMPT_CHOICES = ["box", "point"]
MODELS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/models"


def write_batch_script(
    out_path, dataset_name, model_type, experiment_folder, prompt_choice, dry_run, time_delay, checkpoint,
):
    """Writing batch submission scripts for SAM2 evaluations.
    """
    from run_sam2_evaluations import _get_slurm_template

    batch_script = _get_slurm_template(job_name="sam_mi_evaluation", env_name="super")

    # add delay by a few seconds (to avoid io blocking in container data access)
    batch_script += f"sleep {time_delay}s \n"

    # python script to run.
    pscript = "python evaluate_interactive_3d.py "
    pscript += f"-d {dataset_name} "  # the name of dataset.
    pscript += f"-m {model_type} "  # the choice of model.
    pscript += f"-e {experiment_folder} "  # the path to directory where results will be cached.
    if checkpoint is not None:
        pscript += f"-c {checkpoint} "  # the filepath to model checkpoint.

    if prompt_choice == "box":
        pscript += "--box "  # choice of box prompts for interactive segmentation

    batch_script += pscript

    opath = out_path[:-3] + f"_sam_{dataset_name}_{model_type}_{prompt_choice}.sh"
    with open(opath, "w") as f:
        f.write(batch_script)

    if not dry_run:
        cmd = ["sbatch", opath]
        subprocess.run(cmd)


def submit_to_slurm(dataset_name, experiment_folder, prompt_choice, dry_run):
    """Submit python scripts with given arguments on a compute node via slurm.
    """
    from run_sam2_evaluations import DATASETS_3D, get_batch_script_names

    out_folder = "./gpu_jobs"
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)

    # Create parameter combinations if the user does not request for specific combinations.
    dnames = DATASETS_3D if dataset_name is None else dataset_name
    run_choices = PROMPT_CHOICES if prompt_choice is None else prompt_choice
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

    time_delay = 0
    for dname, rchoice, ckpt_name in itertools.product(dnames, run_choices, checkpoints.keys()):
        ckpt = None if checkpoints[ckpt_name] is None else os.path.join(MODELS_ROOT, checkpoints[ckpt_name])
        mtype = "vit_b"  # NOTE: for the current experiments, we stick to 'vit_b' model.

        write_batch_script(
            out_path=get_batch_script_names(out_folder, script_name="sam_evaluation"),
            dataset_name=dname,
            model_type=mtype,
            experiment_folder=os.path.join(experiment_folder, "3d", "sam", f"{mtype}_{ckpt_name}", dname),
            prompt_choice=rchoice,
            dry_run=dry_run,
            time_delay=time_delay,
            checkpoint=ckpt,
        )
        time_delay += 5  # add time delay by 'n' seconds per job run.


def main(args):
    submit_to_slurm(
        dataset_name=args.dataset_name,
        experiment_folder=args.experiment_folder,
        prompt_choice=args.prompt_choice,
        dry_run=args.dry,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, nargs="*", default=None)
    parser.add_argument(
        "-e", "--experiment_folder", type=str, default="/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam"
    )
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-p", "--prompt_choice", type=str, nargs="*", default=None)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
