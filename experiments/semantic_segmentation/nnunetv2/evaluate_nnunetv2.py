import os
from glob import glob
from tqdm import tqdm

import numpy as np

from medico_sam.evaluation.evaluation import calculate_dice_score

from train_nnunetv2 import DATASET_MAPPING_2D, DATASET_MAPPING_3D, NNUNET_ROOT

from tukra.io import read_image


# These class maps originate from the logic create at  "_common.py"
CLASS_MAPS = {
    # 2d datasets
    "oimhs": {"choroid": 1, "retina": 2, "intraretinal_cysts": 3, "macular_hole": 4},
    "isic": {"skin_lesion": 1},
    "dca1": {"vessel": 1},
    "cbis_ddsm": {"mass": 1},
    "piccolo": {"polyps": 1},
    "hil_toothseg": {"teeth": 1},

    # 3d datasets
    "osic_pulmofib": {"heart": 1, "lung": 2, "trachea": 3},
    "duke_liver": {"liver": 1},
    "oasis": {"gray matter": 1, "thalamus": 2, "white matter": 3, "csf": 4},
    "lgg_mri": {"glioma": 1},
    "leg_3d_us": {"SOL": 1, "GM": 2, "GL": 3},
    "micro_usp": {"prostate": 1},
}


def _evaluate_per_class_dice(gt, prediction, class_maps):
    all_scores = {}
    for cname, cid in class_maps.items():
        per_class_gt = (gt == cid).astype("uint8")
        # Check whether there are objects or it's not relevant for semantic segmentation
        if not len(np.unique(per_class_gt)) > 1:
            continue

        per_class_prediction = (prediction == cid).astype("uint8")
        score = calculate_dice_score(input_=per_class_prediction, target=per_class_gt)

        if cname in all_scores:
            all_scores[cname].extend(score)
        else:
            all_scores[cname] = [score]

    return all_scores


def evaluate_predictions(root_dir, dataset_name, fold, is_3d=False):
    prediction_paths = sorted(
        glob(os.path.join(root_dir, "predictionTs", f"fold_{fold}", "*.nii.gz" if is_3d else "*.tif"))
    )
    gt_paths = sorted(glob(os.path.join(root_dir, "labelsTs", "*.nii.gz" if is_3d else "*.tif")))

    assert len(prediction_paths) == len(gt_paths) and len(prediction_paths) > 0

    dice_scores = []
    for prediction_path, gt_path in tqdm(
        zip(prediction_paths, gt_paths), total=len(gt_paths), desc="Evaluating nnUNet predictions"
    ):
        # Read ground-truth and predictions
        gt = read_image(gt_path)
        prediction = read_image(prediction_path)

        # Convert to integers: we round up because some datasets (eg. CURVAS) have strange labels.
        gt = np.round(gt).astype(int)
        prediction = np.round(prediction).astype(int)

        assert gt.shape == prediction.shape, (gt.shape, prediction.shape)

        score = _evaluate_per_class_dice(gt, prediction, CLASS_MAPS[dataset_name])
        dice_scores.append(score)

    fscores = {}
    for cname in CLASS_MAPS[dataset_name].keys():
        avg_score_per_class = [
            per_image_score.get(cname)[0] for per_image_score in dice_scores if cname in per_image_score
        ]
        fscores[cname] = np.mean(avg_score_per_class)

    print(fscores)


def check_predictions(root_dir, dataset_name, fold, is_3d=False):
    prob_paths = sorted(
        glob(os.path.join(root_dir, "predictionTs", f"fold_{fold}", "*.npz"))
    )

    if len(prob_paths) == 0:
        print(f"No probabilities found for '{dataset_name}'. We can't debug this further.")

    for fpath in prob_paths:
        prob = np.load(fpath)
        print(fpath)
        breakpoint()


def main(args):
    if args.dataset in DATASET_MAPPING_2D:
        dmap_base = DATASET_MAPPING_2D
        is_3d = False
    elif args.dataset in DATASET_MAPPING_3D:
        is_3d = True
        dmap_base = DATASET_MAPPING_3D
    else:
        raise ValueError(args.dataset)

    _, dataset_name = dmap_base[args.dataset]

    root_dir = os.path.join(NNUNET_ROOT, "nnUNet_raw", dataset_name)

    if args.debug:  # Simply visualize the probabilities, if stored.
        check_predictions(root_dir, args.dataset, args.fold, is_3d)
    else:
        evaluate_predictions(root_dir, args.dataset, args.fold, is_3d)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
