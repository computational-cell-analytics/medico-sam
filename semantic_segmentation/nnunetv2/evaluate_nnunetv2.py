# duke_liver: {'liver': 0.914965613366452}


import os
from glob import glob
from tqdm import tqdm

import numpy as np

from medico_sam.evaluation.evaluation import calculate_dice_score

from train_nnunetv2 import DATASET_MAPPING_2D, DATASET_MAPPING_3D, NNUNET_ROOT

from tukra.io import read_image


# These class maps originate from the logic create at  "convert_<DATASET>.py"
CLASS_MAPS = {
    # 2d datasets
    "oimhs": {"choroid": 1, "retina": 2, "intraretinal_cysts": 3, "macular_hole": 4},
    "isic": {"skin_lesion": 1},
    "dca1": {"vessel": 1},
    "cbis_ddsm": {"mass": 1},
    "drive": {"veins": 1},
    "piccolo": {"polyps": 1},
    "siim_acr": {"pneumothorax": 1},
    "hil_toothseg": {"teeth": 1},
    "covid_qu_ex": {"lung": 1},
    # 3d datasets
    "curvas": {"pancreas": 1, "kidney": 2, "liver": 3},
    "osic_oulmofib": {"heart": 1, "lung": 2, "trachea": 3},
    "duke_liver": {"liver": 1},
    "toothfairy": {"mandibular canal": 1},
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


def evaluate_predictions(root_dir, dataset_name, is_3d=False):
    all_predictions = sorted(glob(os.path.join(root_dir, "predictionTs", "*.nii.gz" if is_3d else "*.tif")))
    all_gt = sorted(glob(os.path.join(root_dir, "labelsTs", "*.nii.gz" if is_3d else "*.tif")))

    assert len(all_predictions) == len(all_gt)

    dice_scores = []
    for prediction_path, gt_path in tqdm(zip(all_predictions, all_gt), total=len(all_gt)):
        gt = read_image(gt_path, extension=".nii.gz" if is_3d else ".tif")
        prediction = read_image(prediction_path, extension=".nii.gz" if is_3d else ".tif")

        assert gt.shape == prediction.shape

        class_maps = CLASS_MAPS[dataset_name]
        score = _evaluate_per_class_dice(gt, prediction, class_maps)
        dice_scores.append(score)

    fscores = {}
    for cname in class_maps.keys():
        avg_score_per_class = [
            per_image_score.get(cname)[0] for per_image_score in dice_scores if cname in per_image_score
        ]
        fscores[cname] = np.mean(avg_score_per_class)

    print(fscores)


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

    root_dir = os.path.join(NNUNET_ROOT, "test", dataset_name)
    evaluate_predictions(root_dir, args.dataset, is_3d)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    args = parser.parse_args()
    main(args)
