import os
import sys

import torch

from micro_sam2.evaluation import inference

from sam2_utils import CHECKPOINT_PATHS


sys.path.append("..")


def interactive_segmentation_for_2d_images(
    image_paths,
    gt_paths,
    image_key,
    gt_key,
    model_type,
    backbone,
    device,
    prediction_dir,
    start_with_box=False,
    use_masks=False,
):
    # Interactive segmentation for 2d images using iterative prompting.
    prediction_root = inference.run_interactive_segmentation_2d(
        image_paths=image_paths,
        gt_paths=gt_paths,
        image_key=image_key,
        gt_key=gt_key,
        prediction_dir=prediction_dir,
        model_type=model_type,
        backbone=backbone,
        checkpoint_path=CHECKPOINT_PATHS[backbone][model_type],
        start_with_box_prompt=start_with_box,
        device=device,
        use_masks=use_masks,
    )

    # Evaluating the interactive segmentation results using iterative prompting.
    # TODO: run dice score per semantic class.
    breakpoint()


def main():
    import argparse
    from util import get_dataset_paths

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t")
    parser.add_argument("-b", "--backbone", type=str, default="sam2.0")
    parser.add_argument("-e", "--experiment_folder", type=str, default="experiments")
    parser.add_argument("-p", "--prompt_choice", type=str, default="box")
    parser.add_argument("--use_masks", action="store_true")
    args = parser.parse_args()

    print("Resource Name:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = args.model_type
    backbone = args.backbone
    dataset_name = args.dataset_name

    image_paths, gt_paths, semantic_class_maps = get_dataset_paths(dataset_name=dataset_name, split="test")

    # HACK: testing it on first 200 (or fewer) samples
    image_paths, gt_paths = image_paths[:200], gt_paths[:200]

    prediction_dir = os.path.join(args.experiment_folder, model_type, dataset_name)

    interactive_segmentation_for_2d_images(
        image_paths=image_paths,
        gt_paths=gt_paths,
        image_key=None,
        gt_key=None,
        model_type=model_type,
        backbone=backbone,
        device=device,
        prediction_dir=prediction_dir,
        start_with_box=(args.prompt_choice == "box"),
        use_masks=args.use_masks,
    )


if __name__ == "__main__":
    main()
