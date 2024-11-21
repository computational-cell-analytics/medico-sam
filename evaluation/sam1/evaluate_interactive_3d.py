import os
import sys
from tqdm import tqdm
from pathlib import Path

import pandas as pd

import torch

from micro_sam.evaluation.multi_dimensional_segmentation import segment_slices_from_ground_truth


sys.path.append("..")


def evaluate_interactive_3d(
    image_paths, gt_paths, device, experiment_folder, model_type, prompt_choice, dataset_name, view
):
    """Interactive segmentation scripts for benchmarking micro-sam.
    """
    output_folder = os.path.join(experiment_folder, model_type, dataset_name)
    save_path = os.path.join(output_folder, "results", f"interactive_segmentation_3d_with_{prompt_choice}.csv")
    if os.path.exists(save_path):
        print(
            f"Results for 3d interactive segmentation with '{prompt_choice}' are already stored at '{save_path}'."
        )
        return

    results = []
    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths),
        desc=f"Run interactive segmentation in 3d with '{prompt_choice}'"
    ):
        prediction_dir = os.path.join(output_folder, "interactive_segmentation_3d", f"{prompt_choice}")
        os.makedirs(prediction_dir, exist_ok=True)

        prediction_path = os.path.join(prediction_dir, f"{Path(image_path).stem}.tif")
        if os.path.exists(prediction_path):
            continue

        # TODO: use tukra to load images.
        raw = ...
        labels = ...

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(raw, name="Image")
            v.add_labels(labels, name="Labels")
            napari.run()

        # Segment using label propagation.
        per_vol_result = segment_slices_from_ground_truth(
            volume=raw,
            ground_truth=labels,
            model_type=model_type,
            save_path=prediction_path,
            device=device,
            interactive_seg_mode=prompt_choice,
            min_size=10,
        )
        results.append(per_vol_result)

    results = pd.concat(results)
    results = results.groupby(results.index).mean()
    results.to_csv(save_path)


def main():
    from util import get_dataset_paths, get_default_arguments
    args = get_default_arguments()

    image_paths, gt_paths, _ = get_dataset_paths(dataset_name=args.dataset, split="test")

    # HACK: testing it on first 200 (or fewer) samples
    image_paths, gt_paths = image_paths[:200], gt_paths[:200]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_interactive_3d(
        image_paths=image_paths,
        gt_paths=gt_paths,
        device=device,
        experiment_folder=args.experiment_folder,
        model_type=args.model,
        prompt_choice="box" if args.box else "point",
        dataset_name=args.dataset,
        view=False,
    )


if __name__ == "__main__":
    main()
