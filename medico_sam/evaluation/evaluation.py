import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import List, Union, Dict, Optional

import numpy as np
import pandas as pd
import imageio.v3 as imageio


def calculate_dice_score(input_, target, eps=1e-7):
    numerator = (input_ * target).sum()
    denominator = (input_ * input_).sum() + (target * target).sum()
    score = 2. * (numerator / denominator.clip(min=eps))
    return score


def _run_evaluation_per_semantic_class(gt_paths, prediction_paths, semantic_class_id, verbose=True):
    assert len(gt_paths) == len(prediction_paths)
    dice_scores = []

    for gt_path, pred_path in tqdm(
        zip(gt_paths, prediction_paths), desc="Evaluate predictions", total=len(gt_paths), disable=not verbose
    ):
        assert os.path.exists(gt_path), gt_path
        assert os.path.exists(pred_path), pred_path

        gt = imageio.imread(gt_path)
        gt = (gt == semantic_class_id).astype("uint8")

        pred = imageio.imread(pred_path)
        pred = (pred > 0).astype("uint8")

        dice = calculate_dice_score(input_=pred, target=gt)
        dice_scores.append(dice)

    return dice_scores


def run_evaluation_per_semantic_class(
    gt_paths: List[Union[os.PathLike, str]],
    prediction_paths: List[Union[os.PathLike, str]],
    semantic_class_id: int,
    save_path: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run evaluation for semantic segmentation predictions.

    Args:
        gt_paths: The list of paths to ground-truth images.
        prediction_paths: The list of paths with the instance segmentations to evaluate.
        semantic_class_id: ...
        save_path: Optional path for saving the results.
        verbose: Whether to print the progress.

    Returns:
        A DataFrame that contains the evaluation results.
    """
    assert len(gt_paths) == len(prediction_paths)
    # if a save_path is given and it already exists then just load it instead of running the eval
    if save_path is not None and os.path.exists(save_path):
        return pd.read_csv(save_path)

    dice_scores = _run_evaluation_per_semantic_class(
        gt_paths, prediction_paths, semantic_class_id, verbose=verbose
    )

    results = pd.DataFrame.from_dict({"dice": [np.mean(dice_scores)]})

    if save_path is not None:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        results.to_csv(save_path, index=False)

    return results


def run_evaluation_for_iterative_prompting_per_semantic_class(
    gt_paths: List[Union[os.PathLike, str]],
    prediction_root: Union[os.PathLike, str],
    experiment_folder: Union[os.PathLike, str],
    semantic_class_map: Dict[str, int],
    start_with_box_prompt: bool = False,
    overwrite_results: bool = False,
) -> pd.DataFrame:
    """Run evaluation for iterative prompt-based segmentation predictions per semantic class.

    Args:
        gt_paths: The list of paths to ground-truth images.
        prediction_root: The folder with the iterative prompt-based instance segmentations to evaluate.
        experiment_folder: The folder where all the experiment results are stored.
        semantic_class_map: ...
        start_with_box_prompt: Whether to evaluate on experiments with iterative prompting starting with box.
        overwrite_results: ...

    Returns:
        A DataFrame that contains the evaluation results.
    """
    assert os.path.exists(prediction_root), prediction_root

    for semantic_class_name, semantic_class_id in semantic_class_map.items():
        # Save the results in the experiment folder
        result_folder = os.path.join(experiment_folder, "results", semantic_class_name)
        os.makedirs(result_folder, exist_ok=True)

        csv_path = os.path.join(
            result_folder,
            "iterative_prompts_start_box.csv" if start_with_box_prompt else "iterative_prompts_start_point.csv"
        )

        # Overwrite the previously saved results
        if overwrite_results and os.path.exists(csv_path):
            os.remove(csv_path)

        # If the results have been computed already, it's not needed to re-run it again.
        if os.path.exists(csv_path):
            print(pd.read_csv(csv_path))
            return

        list_of_results = []
        prediction_folders = natsorted(glob(os.path.join(prediction_root, "iteration*")))
        for pred_folder in prediction_folders:
            print("Evaluating", os.path.split(pred_folder)[-1])
            pred_paths = natsorted(glob(os.path.join(pred_folder, semantic_class_name, "*")))
            result = run_evaluation_per_semantic_class(
                gt_paths=gt_paths, prediction_paths=pred_paths, semantic_class_id=semantic_class_id, save_path=None
            )
            list_of_results.append(result)
            print(result)

        res_df = pd.concat(list_of_results, ignore_index=True)
        res_df.to_csv(csv_path)


def run_evaluation_for_semantic_segmentation(
    gt_paths: List[Union[os.PathLike, str]],
    prediction_root: Union[os.PathLike, str],
    experiment_folder: Union[os.PathLike, str],
    semantic_class_map: Dict[str, int],
    overwrite_results: bool = False,
) -> pd.DataFrame:
    """Run evaluation for semantic segmentation predictions per semantic class.

    Args:
        gt_paths: The list of paths to ground-truth images.
        prediction_root: The folder with the iterative prompt-based instance segmentations to evaluate.
        experiment_folder: The folder where all the experiment results are stored.
        semantic_class_map: ...

    Returns:
        A DataFrame that contains the evaluation results.
    """
    assert os.path.exists(prediction_root), prediction_root

    for semantic_class_name, semantic_class_id in semantic_class_map.items():
        # Save the results in the experiment folder
        result_folder = os.path.join(experiment_folder, "results", semantic_class_name)
        os.makedirs(result_folder, exist_ok=True)

        csv_path = os.path.join(result_folder, "semantic_segmentation.csv")

        # Overwrite the previously saved results
        if overwrite_results and os.path.exists(csv_path):
            os.remove(csv_path)

        # If the results have been computed already, it's not needed to re-run it again.
        if os.path.exists(csv_path):
            print(pd.read_csv(csv_path))
            return

        print("Evaluating", prediction_root)
        pred_paths = natsorted(glob(os.path.join(prediction_root, semantic_class_name, "*")))
        result = run_evaluation_per_semantic_class(
            gt_paths=gt_paths, prediction_paths=pred_paths, semantic_class_id=semantic_class_id, save_path=None
        )
        print(result)
        result.to_csv(csv_path)
