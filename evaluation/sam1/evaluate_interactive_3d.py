import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch

from micro_sam.evaluation.multi_dimensional_segmentation import run_multi_dimensional_segmentation_grid_search

from data_utils import _get_data_paths, _load_raw_and_label_volumes


def evaluate_interactive_3d(
    input_path, experiment_folder, model_type, checkpoint_path, prompt_choice, dataset_name, view
):
    """Interactive segmentation scripts for benchmarking micro-sam.
    """
    save_path = os.path.join(experiment_folder, "results", f"interactive_segmentation_3d_with_{prompt_choice}.csv")
    os.makedirs(os.path.join(experiment_folder, "results"), exist_ok=True)
    if os.path.exists(save_path):
        print(
            f"Results for 3d interactive segmentation with '{prompt_choice}' are already stored at '{save_path}'."
        )
        return

    image_paths, gt_paths, _, keys, ensure_channels_first = _get_data_paths(
        path=input_path, dataset_name=dataset_name
    )

    # HACK: testing it on first 200 (or fewer) samples
    image_paths, gt_paths = image_paths[:200], gt_paths[:200]

    prediction_dir = os.path.join(experiment_folder, "interactive_segmentation_3d", f"{prompt_choice}")
    os.makedirs(prediction_dir, exist_ok=True)

    results = []
    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths),
        desc=f"Run interactive segmentation in 3d with '{prompt_choice}'"
    ):
        raw, labels = _load_raw_and_label_volumes(
            raw_path=image_path,
            label_path=gt_path,
            dataset_name=dataset_name,
            channels_first=ensure_channels_first,
            keys=keys,
        )

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(raw, name="Image")
            v.add_labels(labels, name="Labels")
            napari.run()

        # Perform grid-search to get the best parameters
        best_params_path = run_multi_dimensional_segmentation_grid_search(
            volume=raw,
            ground_truth=labels,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            embedding_path=None,
            result_dir=prediction_dir,
            interactive_seg_mode=prompt_choice,
            min_size=10,
            verbose=False,
            evaluation_metric="dice_per_class",
        )

        res_df = pd.read_csv(best_params_path)
        score = res_df["Dice"].iloc[0]
        print(res_df)
        results.append(score)

        # Remove paths to grid search results
        os.remove(best_params_path)
        os.remove(os.path.join(prediction_dir, "all_grid_search_results.csv"))

    results = pd.DataFrame.from_dict([{"results": np.mean(results)}])
    results.to_csv(save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-i", "--input_path", type=str, default="/mnt/vast-nhr/projects/cidas/cca/data")
    parser.add_argument("-m", "--model_type", type=str, default="vit_b")
    parser.add_argument("-e", "--experiment_folder", type=str, default="experiments")
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None)

    parser.add_argument("--box", action="store_true")
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    print("Resource Name:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

    evaluate_interactive_3d(
        input_path=args.input_path,
        experiment_folder=args.experiment_folder,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        prompt_choice="box" if args.box else "points",
        dataset_name=args.dataset,
        view=args.view,
    )


if __name__ == "__main__":
    main()
