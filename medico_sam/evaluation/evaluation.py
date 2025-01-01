import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import List, Union, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from tukra.io import read_image


def calculate_dice_score(input_, target, eps=1e-7):
    numerator = (input_ * target).sum()
    denominator = (input_ * input_).sum() + (target * target).sum()
    score = 2. * (numerator / denominator.clip(min=eps))
    return score


def _run_evaluation_per_semantic_class(
    gt_paths,
    prediction_paths,
    semantic_class_id,
    verbose=True,
    is_multiclass=False,
    keys=None,
    ensure_channels_first=False,
    extension=None,
):
    if keys is None:
        gt_key = pred_key = None
    else:
        gt_key, pred_key = keys

    dice_scores = []
    for gt_path, pred_path in tqdm(
        zip(gt_paths, prediction_paths), total=len(prediction_paths), desc="Evaluate predictions", disable=not verbose
    ):
        assert os.path.exists(gt_path), gt_path
        assert os.path.exists(pred_path), pred_path

        gt = read_image(gt_path, key=gt_key, extension=extension)
        gt = np.round(gt).astype(int)  # convert labels to integer type.
        if ensure_channels_first:
            gt = gt.transpose(2, 0, 1)

        if semantic_class_id is not None:
            gt = (gt == semantic_class_id)
        else:
            gt = (gt > 0)

        # Check whether the image has a valid foreground in the ground truth
        _, counts = np.unique(gt, return_counts=True)
        if len(counts) == 1:
            continue

        pred = read_image(pred_path, key=pred_key, extension=extension)
        pred = np.round(pred).astype(int)  # convert predictions to integer type.

        if is_multiclass:
            if semantic_class_id is None:  # for SPIDER: if None, we return instance segmentation and binarise them.
                pred = (pred > 0)
            else:
                pred = (pred == semantic_class_id)
        else:
            pred = (pred > 0)

        dice = calculate_dice_score(input_=pred.astype("uint8"), target=gt.astype("uint8"))
        dice_scores.append(dice)

    return dice_scores


def run_evaluation_per_semantic_class(
    gt_paths: List[Union[os.PathLike, str]],
    prediction_paths: List[Union[os.PathLike, str]],
    semantic_class_id: int,
    save_path: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = True,
    is_multiclass: bool = False,
    keys: Optional[Tuple[str]] = None,
    ensure_channels_first: bool = False,
    extension: Optional[str] = None,
) -> pd.DataFrame:
    """Run evaluation for semantic segmentation predictions.

    Args:
        gt_paths: The list of paths to ground-truth images.
        prediction_paths: The list of paths with the instance segmentations to evaluate.
        semantic_class_id: The id of the particular semantic class.
        save_path: Optional path for saving the results.
        verbose: Whether to print the progress.
        is_multiclass: Whether the labels have multiple class for semantic segmentation.
        keys: The hierarchy names of inputs in container data format.
        ensure_channels_first: Whether to ensure channels first for multidimensional images.
        extension: The choice of file extension.

    Returns:
        A DataFrame that contains the evaluation results.
    """
    # if a save_path is given and it already exists then just load it instead of running the eval
    if save_path is not None and os.path.exists(save_path):
        return pd.read_csv(save_path)

    dice_scores = _run_evaluation_per_semantic_class(
        gt_paths=gt_paths,
        prediction_paths=prediction_paths,
        semantic_class_id=semantic_class_id,
        verbose=verbose,
        is_multiclass=is_multiclass,
        keys=keys,
        ensure_channels_first=ensure_channels_first,
        extension=extension,
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
    extension: Optional[str] = None,
) -> pd.DataFrame:
    """Run evaluation for iterative prompt-based segmentation predictions per semantic class.

    Args:
        gt_paths: The list of paths to ground-truth images.
        prediction_root: The folder with the iterative prompt-based instance segmentations to evaluate.
        experiment_folder: The folder where all the experiment results are stored.
        semantic_class_map: The semantic segmentation class map.
        start_with_box_prompt: Whether to evaluate on experiments with iterative prompting starting with box.
        overwrite_results: Whether to overwrite results.
        extension: The choice of file extension for predictions.

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
                gt_paths=gt_paths,
                prediction_paths=pred_paths,
                semantic_class_id=semantic_class_id,
                save_path=None,
                extension=extension,
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
    is_multiclass: bool = False,
    overwrite_results: bool = False,
    ensure_channels_first: bool = True,
) -> pd.DataFrame:
    """Run evaluation for semantic segmentation predictions per semantic class.

    Args:
        gt_paths: The list of paths to ground-truth images.
        prediction_root: The folder with the iterative prompt-based instance segmentations to evaluate.
        experiment_folder: The folder where all the experiment results are stored.
        semantic_class_map: The semantic maps available per dataset.
        is_multiclass: Whether the dataset has multiple semantic classes.
        overwrite_results: Whether to overwrite existing results.
        ensure_channels_first: Whether to ensure channels first.

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
        pred_paths = natsorted(
            glob(os.path.join(prediction_root, "all" if is_multiclass else semantic_class_name, "*"))
        )
        result = run_evaluation_per_semantic_class(
            gt_paths=gt_paths,
            prediction_paths=pred_paths,
            semantic_class_id=semantic_class_id,
            save_path=None,
            is_multiclass=is_multiclass,
            ensure_channels_first=ensure_channels_first,
        )
        print(result)
        result.to_csv(csv_path)
