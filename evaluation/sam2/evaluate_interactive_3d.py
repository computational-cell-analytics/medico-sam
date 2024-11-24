import os
import sys

import torch

from micro_sam import util

from micro_sam2.evaluation import inference

# from util import CHECKPOINT_PATHS


sys.path.append("../sam1")


def interactive_segmentation_for_3d_images(
    path,
    dataset_name,
    model_type,
    backbone,
    experiment_folder,
    prompt_choice,
    n_iterations,
    view=False,
    use_masks=False,
):
    from _util import _load_raw_and_label_volumes, _get_data_paths

    min_size = 10
    device = util.get_device()
    image_paths, gt_paths, semantic_maps = _get_data_paths(path=path, dataset_name=dataset_name)

    # HACK: testing it on first 200 (or fewer) samples
    image_paths, gt_paths = image_paths[:200], gt_paths[:200]

    # First stage: Inference
    for i, (image_path, gt_path) in enumerate(zip(image_paths, gt_paths), start=1):
        raw, labels = _load_raw_and_label_volumes(raw_path=image_path, label_path=gt_path)

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(raw, name="Image")
            v.add_labels(labels, name="Labels")
            napari.run()

        # Parameters for inference and evaluation of interactive segmentation in 3d.
        prediction_dir = experiment_folder  # Directory where predictions and evaluated results will be stored.
        start_with_box_prompt = (prompt_choice == "box")  # Whether to start with box / point as in iterative prompting.

        # Interactive segmentation for multi-dimensional images
        prediction_root = inference.run_interactive_segmentation_3d(
            raw=raw,
            labels=labels,
            model_type=model_type,
            backbone=backbone,
            checkpoint_path=CHECKPOINT_PATHS[backbone][model_type],
            start_with_box_prompt=start_with_box_prompt,
            prediction_dir=prediction_dir,
            prediction_fname=os.path.basename(image_path).split(".")[0],
            device=device,
            min_size=min_size,
            n_iterations=n_iterations,  # Total no. of iterations w. iterative prompting for interactive segmentation.
            use_masks=use_masks,
        )

    # Second stage: Evaluate the interactive segmentation for 3d.
    fname_list, label_list = [], []
    for i, (image_path, gt_path) in enumerate(zip(image_paths, gt_paths), start=1):
        raw, labels = _load_raw_and_label_volumes(raw_path=image_path, label_path=gt_path)

        fname_list.append(os.path.basename(image_path).split(".")[0])
        label_list.append(labels)

    ...  # TODO: calculate dice score per semantic class


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("-i", "--input_path", type=str, default="data")
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t")
    parser.add_argument("-b", "--backbone", type=str, default="sam2.0")
    parser.add_argument("-e", "--experiment_folder", type=str, default="experiments")
    parser.add_argument("-p", "--prompt_choice", type=str, default="box")

    parser.add_argument("-iter", "--n_iterations", type=int, default=1)
    parser.add_argument("--use_masks", action="store_true")
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    print("Resource Name:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

    interactive_segmentation_for_3d_images(
        path=args.input_path,
        dataset_name=args.dataset_name,
        model_type=args.model_type,
        backbone=args.backbone,
        experiment_folder=args.experiment_folder,
        prompt_choice=args.prompt_choice,
        n_iterations=args.n_iterations,
        view=args.view,
        use_masks=args.use_masks,
    )


if __name__ == "__main__":
    main()
