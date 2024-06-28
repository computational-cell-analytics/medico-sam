import re
import subprocess
import itertools


CMD = "python submit_all_evaluations.py "
DATASETS = [
    # "idrid", "camus", "uwaterloo_skin", "montgomery", "sega",
    "piccolo", "cbis_ddsm", "dca1", "papila", "jnu-ifm", "siim_acr",
    "isic", "m2caiseg", "btcv",
]
EXPERIMENTS = [
    "vanilla", "generalist_1", "generalist_8", "simplesam_1", "simplesam_8",
    "medsam-self_1", "medsam-self_8", "medsam", "sam-med2d", "sam-med2d-adapter"
]


def run_eval_process(cmd):
    proc = subprocess.Popen(cmd)
    try:
        outs, errs = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        outs, errs = proc.communicate()


def run_specific_experiment(dataset_name, model_type, experiment_set, dry):
    esplits = experiment_set.split("_")
    if len(esplits) > 1:
        experiment_set, gpu = esplits

    cmd = CMD + f"-d {dataset_name} " + f"-m {model_type} " + f"-e {experiment_set}"
    if len(esplits) > 1:
        cmd += f" --gpus {gpu}"

    if dry:
        cmd += " --dry"

    print(f"Running the command: {cmd} \n")
    _cmd = re.split(r"\s", cmd)
    run_eval_process(_cmd)


def run_one_setup(model_choice, all_dataset_list, all_experiment_set_list, dry):
    for (dataset_name, experiment_set) in itertools.product(all_dataset_list, all_experiment_set_list):
        run_specific_experiment(dataset_name, model_choice, experiment_set, dry)
        breakpoint()


def for_medical_generalist(dataset, experiment, dry):
    run_one_setup(
        model_choice="vit_b",
        all_dataset_list=DATASETS if dataset is None else [dataset],
        all_experiment_set_list=EXPERIMENTS if experiment is None else [experiment],
        dry=dry,
    )


def main(args):
    for_medical_generalist(args.dataset, args.experiment, args.dry)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("-e", "--experiment", type=str, default=None)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
