import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/v1/"

EXPERIMENTS = [
    "vanilla",
    "generalistv2-half", "generalistv2-full",  # "generalist_8",
    "simplesam_8", "medsam-self_8", "medsam", "sam-med2d", "sam-med2d-adapter",
]

MODEL = "vit_b"

DATASET_MAPS = {
    "camus": "CAMUS (Cardiac Structures in Echocardipgraphy)",
    "uwaterloo_skin": "UWaterloo Skin (Skin Lesion in Dermoscopy)",
    "montgomery": "Montgomery (Lungs in Chest X-Ray)",
    "sega": "SegA (Aorta in CT)",
    "duke_liver": "DLDS (Liver Segmentation in MRI)",
    "piccolo": "PICCOLO (Polyps in Narrow Band Imaging)",
    "cbis_ddsm": "CBIS DDSM (Lesion Mass in Mammography)",
    "dca1": "DCA1 (Vessels in X-Ray Coronary Angiograms)",
    "papila": "Papila (Optic Disc & Optic Cup in Fundus)",
    "jnu-ifm": "JNU IFM (Pubic Symphysis & Fetal Head in US)",
    "siim_acr": "SIIM ACR (Pneumothorax in Chest X-Ray)",
    "m2caiseg": "m2caisSeg (Tools and Organs in Endoscopy)",
    "toothfairy": "ToothFairy (Mandibular Canal Segmentation in CBCT)",
    "spider": "SPIDER (Lumbar Spine & Vertebrae Segmentation in MRI)",
    "han-seg": "HaN-Seg (Head & Neck Organ Segmentation in CT)",
    "microusp": "MicroUSP (Prostate Segmentation in Micro-Ultrasound)",
}

MODEL_MAPS = {
    "vanilla": "SAM",
    "generalistv2-half": "MedicoSAM*",
    "generalistv2-full": r"$\bf{MedicoSAM*}$",
    "simplesam_8": "Simple FT*",
    "medsam-self_8": "MedSAM*",
    "medsam": "MedSAM",
    "sam-med2d": "FT-SAM",
    "sam-med2d-adapter": "SAM-Med2D",
    "sam2.1": "SAM2 (2.1)",
}


def _get_res_int(semantic_class, experiment_name, dataset_name, ip_results, ib_results, get_all=False):
    if get_all:
        res = {
            "semantic_class": semantic_class,
            "experiment": experiment_name,
            "dataset": dataset_name,
            # point prompts
            "point": ip_results["dice"][0],
            "ip1": ip_results["dice"][1],
            "ip2": ip_results["dice"][2],
            "ip3": ip_results["dice"][3],
            "ip4": ip_results["dice"][4],
            "ip5": ip_results["dice"][5],
            "ip6": ip_results["dice"][6],
            "ip7": ip_results["dice"][7],
            # box prompts
            "box": ib_results["dice"][0],
            "ib1": ib_results["dice"][1],
            "ib2": ib_results["dice"][2],
            "ib3": ib_results["dice"][3],
            "ib4": ib_results["dice"][4],
            "ib5": ib_results["dice"][5],
            "ib6": ib_results["dice"][6],
            "ib7": ib_results["dice"][7],
        }
    else:
        res = {
            "semantic_class": semantic_class,
            "experiment": experiment_name,
            "dataset": dataset_name,
            "point": ip_results["dice"][0],
            "box": ib_results["dice"][0],
            "ip": ip_results["dice"][7],
            "ib": ib_results["dice"][7],
        }

    return pd.DataFrame.from_dict([res])


def _get_mean_df(res_per_class):
    _multiclass = True if len(res_per_class) > 1 else False

    if _multiclass:
        res_per_class = pd.concat(res_per_class, ignore_index=True)
        mean_values = res_per_class.mean(numeric_only=True).to_dict()

        mean_df = pd.DataFrame([{
            'semantic_class': "all",
            'experiment':  res_per_class['experiment'].iloc[0],
            'dataset': res_per_class['dataset'].iloc[0],
            **mean_values
        }])

    else:
        mean_df = pd.concat(res_per_class, ignore_index=True)

    return mean_df


def _get_vanilla_sam_res_with_mask(dataset_name, get_all=False):
    res_per_class = []
    base_dir = "/mnt/vast-nhr/projects/cidas/cca/experiments/v2"
    for res_dir in glob(os.path.join(base_dir, "vanilla", dataset_name, MODEL, "results", "*")):
        semantic_class = os.path.split(res_dir)[-1]

        ib_results = pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_box.csv"))
        ip_results = pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_point.csv"))

        res_per_class.append(
            _get_res_int(semantic_class, "vanilla", dataset_name, ip_results, ib_results, get_all)
        )

    return _get_mean_df(res_per_class)


def _get_sam_results_per_dataset_per_class(dataset_name, experiment_name, get_all=False):
    res_per_class = []

    root_dir = ROOT
    if experiment_name.startswith("generalist"):
        root_dir = root_dir.replace("v1", "v3")  # NOTE: Switches all results to new inference outcomes.

    for res_dir in glob(os.path.join(root_dir, experiment_name, dataset_name, MODEL, "results", "*")):
        semantic_class = os.path.split(res_dir)[-1]

        ib_results = pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_box.csv"))
        ip_results = pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_point.csv"))

        res_per_class.append(
            _get_res_int(semantic_class, experiment_name, dataset_name, ip_results, ib_results, get_all)
        )

    return _get_mean_df(res_per_class)


def _get_sam2_results_per_dataset_per_class(
    dataset_name, backbone, mask_dir="iterative_prompting_with_mask", model="hvit_b", get_all=False,
):
    res_per_class = []
    base_dir = "/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam/2d"
    for res_dir in glob(os.path.join(base_dir, backbone, model, dataset_name, "results", "*")):
        semantic_class = os.path.split(res_dir)[-1]

        ib_results = pd.read_csv(os.path.join(res_dir, mask_dir, "iterative_prompts_start_box.csv"))
        ip_results = pd.read_csv(os.path.join(res_dir, mask_dir, "iterative_prompts_start_point.csv"))

        res_per_class.append(_get_res_int(semantic_class, backbone, dataset_name, ip_results, ib_results, get_all))

    return _get_mean_df(res_per_class)


def _get_results_per_dataset(dataset_name, get_all=False, use_masks=True):
    res_per_dataset = []
    for experiment_name in EXPERIMENTS:
        # NOTE: We 'use_masks' for iterative prompting for vanilla SAM1 model.
        if experiment_name == "vanilla" and use_masks:
            res = _get_vanilla_sam_res_with_mask(dataset_name, get_all=get_all)
        else:
            res = _get_sam_results_per_dataset_per_class(dataset_name, experiment_name, get_all=get_all)

        res_per_dataset.append(res)

    # Get SAM2 results
    # NOTE: It's hard-coded at the moment in 'mask_dir' argument that uses "with_masks" for iterative prompting.
    res_per_dataset.append(_get_sam2_results_per_dataset_per_class(dataset_name, "sam2.1", get_all=get_all))
    # res_per_dataset.append(_get_sam2_results_per_dataset_per_class(dataset_name, "sam2.0", get_all=get_all))

    res_per_dataset = pd.concat(res_per_dataset, ignore_index=True)
    return res_per_dataset


def _make_per_experiment_plots(dataframes, datasets):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(32, 32))
    axes = axes.flatten()

    bar_width = 0.2
    for i, df in enumerate(dataframes):
        _order = [
            "vanilla", "medsam", "sam-med2d", "sam-med2d-adapter", "medsam-self_8", "simplesam_8",
            "sam2.1",  # "sam2.0",
            "generalistv2-half", "generalistv2-full",  # "generalist_8",
        ]
        df['experiment'] = pd.Categorical(df['experiment'], categories=_order, ordered=True)
        df = df.sort_values('experiment')

        ref = df[df['experiment'] == 'vanilla'].iloc[0]

        df = df[df['experiment'] != 'vanilla']

        df_diff = df.copy()
        df_diff['point_diff'] = df['point'] - ref['point']
        df_diff['box_diff'] = df['box'] - ref['box']
        df_diff['ip_diff'] = df['ip'] - ref['ip']
        df_diff['ib_diff'] = df['ib'] - ref['ib']

        r1 = np.arange(len(df))
        r2 = [x + bar_width for x in r1]
        r3 = [x + 2 * bar_width for x in r1]
        r4 = [x + 3 * bar_width for x in r1]

        axes[i].bar(r1, df_diff['point_diff'], color='#7CCBA2', width=bar_width, edgecolor='grey', label='Point')
        axes[i].bar(r2, df_diff['box_diff'], color='#FCDE9C', width=bar_width, edgecolor='grey', label='Box')
        axes[i].bar(r3, df_diff['ip_diff'], color='#045275', width=bar_width, edgecolor='grey', label=r"I$_{P}$")
        axes[i].bar(r4, df_diff['ib_diff'], color='#90477F', width=bar_width, edgecolor='grey', label=r"I$_{B}$")

        max_val = max(df_diff[['point_diff', 'box_diff', 'ip_diff', 'ib_diff']].values.flatten())
        min_val = min(df_diff[['point_diff', 'box_diff', 'ip_diff', 'ib_diff']].values.flatten())

        axes[i].axhspan(0, max_val, facecolor='lightgreen', alpha=0.2)
        axes[i].axhspan(min_val, 0, facecolor='lightcoral', alpha=0.2)

        _xticklabels = [MODEL_MAPS[_exp] for _exp in df["experiment"]]
        tick_positions = [r + 3 * bar_width for r in range(len(df))]
        axes[i].set_xticks(tick_positions)
        axes[i].set_xticklabels(_xticklabels, rotation=45, ha='right', fontsize=16)
        axes[i].tick_params(axis='y', labelsize=14)

        axes[i].set_title(f'{DATASET_MAPS[datasets[i]]}', fontsize=16)
        axes[i].legend()

    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.legend().remove()

    fig.legend(all_lines, all_labels, loc="lower center", ncols=4, bbox_to_anchor=(0.5, 0), fontsize=24)

    plt.text(
        x=-32.75, y=1, s="Relative Dice Similarity Coefficient (compared to SAM)",
        rotation=90, fontweight="bold", fontsize=24
    )

    plt.subplots_adjust(top=0.95, bottom=0.075, right=0.95, left=0.05, hspace=0.3, wspace=0.2)
    plt.savefig("./fig_3_interactive_segmentation_per_dataset.png", bbox_inches="tight", dpi=600)
    plt.savefig("./fig_3_interactive_segmentation_per_dataset.svg", bbox_inches="tight", dpi=600)
    plt.close()


def _make_per_model_average_plots(dataframes):
    all_data = pd.concat(dataframes, ignore_index=True)
    desired_experiments = [
        "vanilla", "generalistv2-full", "medsam", "sam2.1",  # "sam2.0"
    ]
    filtered_data = all_data[all_data['experiment'].isin(desired_experiments)]

    grouped_data = filtered_data.groupby('experiment')[['point', 'box', 'ip', 'ib']].mean().reset_index()

    _order = [
        "vanilla", "sam2.1", "medsam", "generalistv2-full"
    ]
    grouped_data['experiment'] = pd.Categorical(grouped_data['experiment'], categories=_order, ordered=True)
    grouped_data = grouped_data.sort_values('experiment')

    experiments = grouped_data['experiment']

    metrics = ['point', 'box', 'ip', 'ib']
    color_map = ['#7CCBA2', '#FCDE9C', '#045275', '#90477F']
    label_map = ["Point", "Box", r"I$_{P}$", r"I$_{B}$"]

    x = np.arange(len(experiments))
    width = 0.2

    fig, ax = plt.subplots(figsize=(20, 15))
    for i, (metric, color, label) in enumerate(zip(metrics, color_map, label_map)):
        ax.bar(x + i * width, grouped_data[metric], width, label=label, color=color, edgecolor='grey')

    ax.set_ylabel('Dice Similarity Coefficient', fontsize=20, fontweight="bold")
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    _xticklabels = [MODEL_MAPS[_exp] for _exp in experiments]
    ax.set_xticklabels(_xticklabels, fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend()

    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.legend().remove()

    fig.legend(all_lines, all_labels, loc="upper center", ncols=4, bbox_to_anchor=(0.29, 0.875), fontsize=18)

    plt.title("Interactive Segmentation (2D)", fontsize=24, fontweight="bold")
    plt.savefig("./fig_1b_interactive_segmentation_2d_average.png", bbox_inches="tight")
    plt.savefig("./fig_1b_interactive_segmentation_2d_average.svg", bbox_inches="tight")
    plt.close()


def _make_full_iterative_prompting_average_plots(dataframes):
    combined_df = pd.concat(dataframes, ignore_index=True)

    numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns.insert(0, 'experiment')

    avg_df = combined_df[numeric_columns].groupby('experiment').mean().reset_index()

    _order = [
        "vanilla", "sam2.1", "medsam", "sam-med2d", "sam-med2d-adapter", "medsam-self_8", "simplesam_8",
        "generalistv2-half", "generalistv2-full",  # "generalist_8"
    ]
    avg_df['experiment'] = pd.Categorical(avg_df['experiment'], categories=_order, ordered=True)
    avg_df = avg_df.sort_values('experiment')

    experiments = avg_df['experiment']
    point_values = avg_df['point']
    box_values = avg_df['box'] - avg_df['point']

    ip_columns = [col for col in avg_df.columns if col.startswith('ip')]
    ib_columns = [col for col in avg_df.columns if col.startswith('ib')]

    ip_values = avg_df[ip_columns]
    ib_values = avg_df[ib_columns].values - avg_df[ip_columns].values

    fig, ax = plt.subplots(figsize=(30, 15))

    bar_width = 0.11
    index = np.arange(len(experiments))

    num_colors = len(ip_columns)

    bcolors = [mcolors.to_rgba('#FCDE9C', alpha=(i + 2) / (num_colors + 1)) for i in range(num_colors - 1)][::-1]
    bcolors += ["#90477F"]
    pcolors = [mcolors.to_rgba('#7CCBA2', alpha=(i + 2) / (num_colors + 1)) for i in range(num_colors - 1)][::-1]
    pcolors += ["#045275"]

    ax.bar(index, point_values, bar_width, color=pcolors[0], label='Point', edgecolor="grey")
    ax.bar(index, box_values, bar_width, bottom=point_values, color=bcolors[0], label='Box', edgecolor="grey")

    for i in range(num_colors):
        ip_i = ip_values.iloc[:, i]
        ib_i = ib_values[:, i]
        ax.bar(
            index + bar_width * (i + 1), ip_i, bar_width,
            color=pcolors[i], label=r"I$_{P}$", edgecolor="grey",
        )
        ax.bar(
            index + bar_width * (i + 1), ib_i, bar_width, bottom=ip_i,
            color=bcolors[i], label=r"I$_{B}$", edgecolor="grey"
        )

    ax.set_ylabel('Dice Similarity Coefficient', fontsize=16, fontweight="bold")
    ax.set_xticks(index + bar_width * (num_colors / 2))
    _xticklabels = [MODEL_MAPS[_exp] for _exp in experiments]
    ax.set_xticklabels(_xticklabels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)

    handles, labels = ax.get_legend_handles_labels()
    simplified_handles = [handles[0], handles[1], handles[-2], handles[-1]]
    simplified_labels = [labels[0], labels[1], labels[-2], labels[-1]]
    fig.legend(
        simplified_handles, simplified_labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 0.875), fontsize=16
    )

    plt.savefig("./fig_3_interactive_segmentation_average_iterative_prompting.png", bbox_inches="tight")
    plt.savefig("./fig_3_interactive_segmentation_average_iterative_prompting.svg", bbox_inches="tight")
    plt.close()


def _figure_1b():
    # for point, box, ip and ib
    results = []
    for dataset_name in list(DATASET_MAPS.keys()):
        res = _get_results_per_dataset(dataset_name=dataset_name, use_masks=True)
        results.append(res)

    _make_per_model_average_plots(results)


def _figure_3a():
    # for all iterations in iterative prompting
    results = []
    for dataset_name in list(DATASET_MAPS.keys()):
        res = _get_results_per_dataset(dataset_name=dataset_name, get_all=True, use_masks=True)
        results.append(res)

    _make_full_iterative_prompting_average_plots(results)


def _figure_3b():
    # for point, box, ip and ib
    results = []
    for dataset_name in list(DATASET_MAPS.keys()):
        res = _get_results_per_dataset(dataset_name=dataset_name, use_masks=True)
        results.append(res)

    _make_per_experiment_plots(results, list(DATASET_MAPS.keys()))


def main():
    _figure_1b()
    # _figure_3a()
    # _figure_3b()


if __name__ == "__main__":
    main()
