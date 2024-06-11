import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Dict, Optional

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy


def _run_evaluation_per_semantic_class(gt_paths, prediction_paths, semantic_class_id, verbose=True):
    assert len(gt_paths) == len(prediction_paths)
    msas, sa50s, sa75s = [], [], []

    for gt_path, pred_path in tqdm(
        zip(gt_paths, prediction_paths), desc="Evaluate predictions", total=len(gt_paths), disable=not verbose
    ):
        assert os.path.exists(gt_path), gt_path
        assert os.path.exists(pred_path), pred_path

        gt = imageio.imread(gt_path)
        gt = (gt == semantic_class_id).astype("uint8")

        pred = imageio.imread(pred_path)

        msa, scores = mean_segmentation_accuracy(pred, gt, return_accuracies=True)
        sa50, sa75 = scores[0], scores[5]
        msas.append(msa), sa50s.append(sa50), sa75s.append(sa75)

    return msas, sa50s, sa75s


def run_evaluation_per_semantic_class(
    gt_paths: List[Union[os.PathLike, str]],
    prediction_paths: List[Union[os.PathLike, str]],
    semantic_class_id: int,
    save_path: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run evaluation for instance segmentation predictions.

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

    msas, sa50s, sa75s = _run_evaluation_per_semantic_class(
        gt_paths, prediction_paths, semantic_class_id, verbose=verbose
    )

    results = pd.DataFrame.from_dict({
        "msa": [np.mean(msas)],
        "sa50": [np.mean(sa50s)],
        "sa75": [np.mean(sa75s)],
    })

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
        prediction_folders = sorted(glob(os.path.join(prediction_root, "iteration*")))
        for pred_folder in prediction_folders:
            print("Evaluating", os.path.split(pred_folder)[-1])
            pred_paths = sorted(glob(os.path.join(pred_folder, semantic_class_name, "*")))
            result = run_evaluation_per_semantic_class(gt_paths=gt_paths, prediction_paths=pred_paths, save_path=None)
            list_of_results.append(result)
            print(result)

        res_df = pd.concat(list_of_results, ignore_index=True)
        res_df.to_csv(csv_path)
