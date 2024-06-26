import os
import itertools
from glob import glob

import pandas as pd


ROOT = "/scratch/share/cidas/cca/experiments/v1/"
DATASETS = [
    "idrid", "camus", "uwaterloo_skin", "montgomery", "sega"
]
EXPERIMENTS = [
    "vanilla", "generalist_1", "generalist_8", "simplesam_1", "simplesam_8",
    "medsam-self_1", "medsam-self_8", "medsam", "sam-med2d", "sam-med2d-adapter"
]
MODEL = "vit_b"


def _get_results_per_dataset(dataset_name, experiment_name):
    for res_dir in glob(os.path.join(ROOT, experiment_name, dataset_name, MODEL, "results", "*")):
        semantic_class = os.path.split(res_dir)[-1]

        try:
            ib_results = pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_box.csv"))
            ip_results = pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_point.csv"))
        except FileNotFoundError:
            continue

        res = {
            "semantic_class": semantic_class,
            "experiment": experiment_name,
            "dataset": dataset_name,
            "point": ip_results["dice"][0],
            "box": ib_results["dice"][0],
            "ip": ip_results["dice"][7],
            "ib": ib_results["dice"][7],
        }

        print(res)

        breakpoint()


def main():
    for (dataset_name, experiment_name) in itertools.product(DATASETS, EXPERIMENTS):
        _get_results_per_dataset(dataset_name=dataset_name, experiment_name=experiment_name)


main()
