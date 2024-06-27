import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = "/scratch/share/cidas/cca/experiments/v1/"
DATASETS = [
    "idrid", "camus", "uwaterloo_skin", "montgomery", "sega"
]
EXPERIMENTS = [
    "vanilla", "generalist_8", "simplesam_8", "medsam-self_8", "medsam", "sam-med2d", "sam-med2d-adapter"
    # "generalist_1", "simplesam_1", "medsam-self_1",
]
MODEL = "vit_b"


def _get_results_per_dataset_per_class(dataset_name, experiment_name):
    res_per_class = []
    for res_dir in glob(os.path.join(ROOT, experiment_name, dataset_name, MODEL, "results", "*")):
        semantic_class = os.path.split(res_dir)[-1]

        ib_results = pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_box.csv"))
        ip_results = pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_point.csv"))

        res = {
            "semantic_class": semantic_class,
            "experiment": experiment_name,
            "dataset": dataset_name,
            "point": ip_results["dice"][0],
            "box": ib_results["dice"][0],
            "ip": ip_results["dice"][7],
            "ib": ib_results["dice"][7],
        }

        res_per_class.append(pd.DataFrame.from_dict([res]))

    res_per_class = pd.concat(res_per_class, ignore_index=True)
    return res_per_class


def _get_results_per_dataset(dataset_name):
    res_per_dataset = []
    for experiment_name in EXPERIMENTS:
        res_per_dataset.append(_get_results_per_dataset_per_class(dataset_name, experiment_name))

    res_per_dataset = pd.concat(res_per_dataset, ignore_index=True)
    return res_per_dataset


def _make_plots(dataframes, datasets):
    # Create subplots in a 3x3 grid
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    axes = axes.flatten()

    # Calculate the differences relative to the "vanilla" experiment
    bar_width = 0.2  # Width of the bars
    for i, df in enumerate(dataframes):
        # Reference row (vanilla experiment)
        ref = df[df['experiment'] == 'vanilla'].iloc[0]

        df = df[df['experiment'] != 'vanilla']

        # Calculate differences
        df_diff = df.copy()
        df_diff['point_diff'] = df['point'] - ref['point']
        df_diff['box_diff'] = df['box'] - ref['box']
        df_diff['ip_diff'] = df['ip'] - ref['ip']
        df_diff['ib_diff'] = df['ib'] - ref['ib']

        # Positions for bars
        r1 = np.arange(len(df))  # Positions for point_diff bars
        r2 = [x + bar_width for x in r1]  # Positions for box_diff bars
        r3 = [x + 2*bar_width for x in r1]  # Positions for ip_diff bars
        r4 = [x + 3*bar_width for x in r1]  # Positions for ib_diff bars

        # Plot bars for point_diff, box_diff, ip_diff, and ib_diff
        axes[i].bar(r1, df_diff['point_diff'], color='b', width=bar_width, edgecolor='grey', label='point_diff')
        axes[i].bar(r2, df_diff['box_diff'], color='lightblue', width=bar_width, edgecolor='grey', label='box_diff')
        axes[i].bar(r3, df_diff['ip_diff'], color='g', width=bar_width, edgecolor='grey', label='ip_diff')
        axes[i].bar(r4, df_diff['ib_diff'], color='lightgreen', width=bar_width, edgecolor='grey', label='ib_diff')

        # Adjust x-axis ticks and labels
        axes[i].set_xticks([r + 1.5 * bar_width for r in range(len(df))])
        axes[i].set_xticklabels(df['experiment'], rotation=45, ha='right')

        # Set labels and title
        axes[i].set_xlabel('Experiment', fontweight='bold')
        axes[i].set_ylabel('Difference from Vanilla', fontweight='bold')
        axes[i].set_title(f'{datasets[i]}')

        # Add legend
        axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig("./test.png")


def main():
    results = []
    for dataset_name in DATASETS:
        res = _get_results_per_dataset(dataset_name=dataset_name)
        results.append(res)

    _make_plots(results, DATASETS)
    breakpoint()


main()
