import os
import sys
from glob import glob
from natsort import natsorted

import pandas as pd

import torch

from micro_sam2.evaluation import inference

from medico_sam.evaluation.evaluation import run_evaluation_per_semantic_class

from sam2_utils import CHECKPOINT_PATHS


sys.path.append("..")


def interactive_segmentation_for_2d_images(
    image_paths,
    gt_paths,
    image_key,
    gt_key,
    semantic_class_maps,
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
    result_folder = os.path.join(prediction_dir, "results")
    os.makedirs(result_folder, exist_ok=True)

    for semantic_class_name, semantic_class_id in semantic_class_maps.items():
        # Save the results in the experiment folder
        csv_path = os.path.join(
            result_folder, "iterative_prompts_start_box.csv" if start_with_box else "iterative_prompts_start_point.csv"
        )

        # If the results have been computed already, it's not needed to re-run it again.
        # if os.path.exists(csv_path):
        #     print(pd.read_csv(csv_path))
        #     return

        list_of_results = []
        prediction_folders = natsorted(glob(os.path.join(prediction_root, "iteration*")))
        for pred_folder in prediction_folders:
            print("Evaluating", os.path.split(pred_folder)[-1])
            pred_paths = natsorted(glob(os.path.join(pred_folder, "*")))
            result = run_evaluation_per_semantic_class(
                gt_paths=gt_paths, prediction_paths=pred_paths, semantic_class_id=semantic_class_id,
            )
            list_of_results.append(result)
            print(result)

        res_df = pd.concat(list_of_results, ignore_index=True)
        res_df.to_csv(csv_path)


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
        semantic_class_maps=semantic_class_maps,
        model_type=model_type,
        backbone=backbone,
        device=device,
        prediction_dir=prediction_dir,
        start_with_box=(args.prompt_choice == "box"),
        use_masks=args.use_masks,
    )


if __name__ == "__main__":
    main()
