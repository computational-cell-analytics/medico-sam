import os
from glob import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam/3d"

DATASETS = {
    "lgg_mri": "LGG MRI (Low-Grade Glioma in MRI)",
    # "leg_3d_us": "LEG 3D US",
    "microusp": "MicroUSP (Prostate in Micro-Ultrasound)",
    "duke_liver": "Duke Liver (Liver in MRI)",
}

MODEL_MAPS = {
    "sam": "SAM",
    "simplesam": "Simple FT",
    "medsam": "MedSAM",
    "sam2.0": "SAM2 (2.0)",
    "sam2.1": "SAM2 (2.1)",
    "medico-sam-8g": r"$\bf{MedicoSAM}$",
}


def _get_sam2_results_per_dataset(dataset_name):
    res_list = []
    for bpath in glob(os.path.join(ROOT, "*")):
        backbone = os.path.basename(bpath)
        for res_path in glob(os.path.join(bpath, "hvit_b", dataset_name, "results", "**", "*.csv")):
            res = pd.read_csv(res_path)
            roi = res.columns[-1]
            res_dict = {
                "backbone": backbone,
                "prompt_choice": Path(res_path).stem.split("_")[-1],
                "type": roi,
                "score": res.iloc[0][roi]
            }
            res_list.append(pd.DataFrame.from_dict([res_dict]))

    res_df = pd.concat(res_list, ignore_index=True)
    return res_df


def _get_sam_results_per_dataset(dataset_name):
    res_list = []
    for res_path in glob(os.path.join(ROOT, "sam", "*", dataset_name, "results", "*.csv")):
        res = pd.read_csv(res_path)
        roi = res.columns[-1]

        # Update the prompt names for consistency.
        pchoice = Path(res_path).stem.split("_")[-1]
        if pchoice == "points":
            pchoice = pchoice[:-1]

        res_dict = {
            "backbone": res_path.rsplit("/")[-4][6:],
            "prompt_choice": pchoice,
            "type": roi,
            "score": res.iloc[0][roi]
        }
        res_list.append(pd.DataFrame.from_dict([res_dict]))

    res_df = pd.concat(res_list, ignore_index=True)
    return res_df


def _get_plots():
    fig, ax = plt.subplots(1, 3, figsize=(30, 15))

    for i, (dname, dmap) in enumerate(DATASETS.items()):
        # Extract results per dataset
        df1 = _get_sam_results_per_dataset(dname)
        df2 = _get_sam2_results_per_dataset(dname)

        df = pd.concat([df1, df2], ignore_index=True)

        methods = MODEL_MAPS.keys()
        labels = [MODEL_MAPS[m] for m in methods]

        scores = []
        for _method in methods:
            base_res = df[df['backbone'] == _method]
            res = {"method": _method}

            if "box" in base_res['prompt_choice'].tolist():
                res["box"] = df[(df['backbone'] == _method) & (df['prompt_choice'] == 'box')]['score'].iloc[0]
            else:
                res["box"] = 0

            if "point" in base_res['prompt_choice'].tolist():
                res["point"] = df[(df['backbone'] == _method) & (df['prompt_choice'] == 'point')]['score'].iloc[0]
            else:
                res["point"] = 0

            scores.append(res)

        bar_width = 0.4
        x = range(len(methods))

        ax[i].bar(
            [pos - bar_width / 2 for pos in x], [s.get("box") for s in scores],
            width=bar_width, label='Box', color="#FCDE9C", edgecolor="grey",
        )
        ax[i].bar(
            [pos + bar_width / 2 for pos in x], [s.get("point") for s in scores],
            width=bar_width, label='Point', color="#7CCBA2", edgecolor="grey",
        )

        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
        ax[i].tick_params(axis='y', labelsize=16)  # Set y-tick label size
        ax[i].set_title(dmap, fontweight="bold", fontsize=16)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=16, ncol=2)

    plt.text(
        x=-15.4, y=0.24, s="Dice Similarity Coefficient", rotation=90, fontweight="bold", fontsize=20
    )

    plt.subplots_adjust(top=0.9, bottom=0.125, right=0.95, left=0.05, wspace=0.1)
    plt.savefig("./fig_5_interactive_segmentation_3d_per_dataset.png", bbox_inches="tight")
    plt.savefig("./fig_5_interactive_segmentation_3d_per_dataset.svg", bbox_inches="tight")


def main():
    _get_plots()


if __name__ == "__main__":
    main()
