import os
from tqdm import tqdm
from pathlib import Path

import pandas as pd

import torch

from micro_sam.evaluation.multi_dimensional_segmentation import segment_slices_from_ground_truth

from _util import _get_data_paths, _load_raw_and_label_volumes


def evaluate_interactive_3d(
    input_path, device, experiment_folder, model_type, prompt_choice, dataset_name, view
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

    image_paths, gt_paths, semantic_maps = _get_data_paths(path=input_path, dataset_name=dataset_name)

    # HACK: testing it on first 200 (or fewer) samples
    image_paths, gt_paths = image_paths[:200], gt_paths[:200]

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

        raw, labels = _load_raw_and_label_volumes(raw_path=image_path, label_path=gt_path)

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(raw, name="Image")
            v.add_labels(labels, name="Labels")
            napari.run()

        # Segment using label propagation.
        _, segmentation = segment_slices_from_ground_truth(
            volume=raw,
            ground_truth=labels,
            model_type=model_type,
            save_path=prediction_path,
            device=device,
            interactive_seg_mode=prompt_choice,
            min_size=10,
            return_segmentation=True,
            verbose=True,
        )

        breakpoint()

        per_vol_result = {}
        results.append(per_vol_result)

    results = pd.concat(results)
    results = results.groupby(results.index).mean()
    results.to_csv(save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-i", "--input_path", type=str, default="data")
    parser.add_argument("-m", "--model_type", type=str, default="vit_b")
    parser.add_argument("-e", "--experiment_folder", type=str, default="experiments")

    parser.add_argument("--box", action="store_true")
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    print("Resource Name:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate_interactive_3d(
        input_path=args.input_path,
        device=device,
        experiment_folder=args.experiment_folder,
        model_type=args.model_type,
        prompt_choice="box" if args.box else "point",
        dataset_name=args.dataset,
        view=args.view,
    )


if __name__ == "__main__":
    main()
