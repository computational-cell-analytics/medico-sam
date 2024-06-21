import re
import subprocess


CMD = "python submit_all_evaluations.py "


def run_eval_process(cmd):
    proc = subprocess.Popen(cmd)
    try:
        outs, errs = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        outs, errs = proc.communicate()


def run_specific_experiment(dataset_name, model_type, experiment_set, gpu):
    cmd = CMD + f"-d {dataset_name} " + f"-m {model_type} " + f"-e {experiment_set} " + f"--gpus {gpu}"
    print(f"Running the command: {cmd} \n")
    _cmd = re.split(r"\s", cmd)
    run_eval_process(_cmd)


def run_one_setup(model_choice, all_dataset_list, all_experiment_set_list, n_gpus):
    for dataset_name in all_dataset_list:
        for experiment_set in all_experiment_set_list:
            for gpu in n_gpus:
                run_specific_experiment(dataset_name, model_choice, experiment_set, gpu)


def for_medical_generalist():
    n_gpus = [2]
    run_one_setup(
        model_choice="vit_b",
        all_dataset_list=["idrid", "camus", "uwaterloo_skin", "montgomery", "sega"],
        all_experiment_set_list=["vanilla", "generalist", "medsam"],
        n_gpus=n_gpus,
    )


def main():
    for_medical_generalist()


if __name__ == "__main__":
    main()
