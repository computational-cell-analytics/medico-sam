import os
from tqdm import tqdm

import h5py

import torch

from micro_sam.evaluation.multi_dimensional_segmentation import segment_slices_from_ground_truth

from data_utils import _get_data_paths, _load_raw_and_label_volumes


def evaluate_interactive_3d(
    input_path, experiment_folder, model_type, checkpoint_path, dataset_name
):
    """Interactive segmentation scripts for benchmarking micro-sam.
    """
    save_path = os.path.join(experiment_folder, "results", "interactive_segmentation_3d.csv")
    os.makedirs(os.path.join(experiment_folder, "results"), exist_ok=True)
    if os.path.exists(save_path):
        print(f"Results for 3d interactive segmentation already stored at '{save_path}'.")
        return

    image_paths, gt_paths, _, keys, ensure_channels_first = _get_data_paths(
        path=input_path, dataset_name=dataset_name
    )

    prediction_dir = os.path.join(experiment_folder, "interactive_segmentation_3d", "box")
    os.makedirs(prediction_dir, exist_ok=True)

    counter = 0
    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths), desc="Run interactive segmentation in 3d'"
    ):
        raw, labels = _load_raw_and_label_volumes(
            raw_path=image_path,
            label_path=gt_path,
            dataset_name=dataset_name,
            channels_first=ensure_channels_first,
            keys=keys,
        )

        res, segmentation = segment_slices_from_ground_truth(
            volume=raw,
            ground_truth=labels,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            save_path=os.path.join(prediction_dir, f"test_{counter}.tif"),
            iou_threshold=0.7,
            projection="box",  # "mask", "points", "box", "points_and_mask", "single_point"
            interactive_seg_mode="box",
            return_segmentation=True,
            evaluation_metric="dice_per_class",
        )

        counter += 1

        print(res, counter)

        with h5py.File(os.path.join(experiment_folder, f"{dataset_name}_{counter}.h5"), "w") as f:
            f.create_dataset("raw", data=raw)
            f.create_dataset("labels", data=labels, compression="gzip")
            f.create_dataset("segmentation", data=segmentation, compression="gzip")

        # if counter > 0:
        #     break


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-i", "--input_path", type=str, default="/mnt/vast-nhr/projects/cidas/cca/data")
    parser.add_argument("-m", "--model_type", type=str, default="vit_b")
    parser.add_argument("-e", "--experiment_folder", type=str, default="experiments")
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    print("Resource Name:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

    evaluate_interactive_3d(
        input_path=args.input_path,
        experiment_folder=args.experiment_folder,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
