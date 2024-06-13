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


def run_specific_experiment(dataset_name, model_type, experiment_set):
    cmd = CMD + f"-d {dataset_name} " + f"-m {model_type} " + f"-e {experiment_set} "
    print(f"Running the command: {cmd} \n")
    _cmd = re.split(r"\s", cmd)
    run_eval_process(_cmd)


def run_one_setup(model_choice, all_dataset_list, all_experiment_set_list):
    for dataset_name in all_dataset_list:
        for experiment_set in all_experiment_set_list:
            run_specific_experiment(dataset_name, model_choice, experiment_set)


def for_medical_generalist():
    run_one_setup(
        model_choice="vit_b",
        all_dataset_list=["covid_if"],
        all_experiment_set_list=["vanilla", "generalist", "medsam"],
    )


def main():
    for_medical_generalist()


if __name__ == "__main__":
    main()
