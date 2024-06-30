import os
from glob import glob
from tqdm import tqdm

import numpy as np
import imageio.v3 as imageio

from medico_sam.evaluation.evaluation import calculate_dice_score

from train_nnunetv2 import DATASET_MAPPING_2D, DATASET_MAPPING_3D, NNUNET_ROOT


# These class maps originate from the logic create at  "convert_<DATASET>.py"
CLASS_MAPS = {
    "oimhs": {"choroid": 1, "retina": 2, "intraretinal_cysts": 3, "macular_hole": 4},
    "isic": {"skin_lesion": 1},
    "dca1": {"vessel": 1},
    "cbis_ddsm": {"mass": 1},
    "piccolo": {"polyp": 1},
    "drive": {"vessel": 1},
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


def evaluate_predictions(root_dir, dataset_name):
    all_predictions = sorted(glob(os.path.join(root_dir, "predictionTs", "*.tif")))
    all_gt = sorted(glob(os.path.join(root_dir, "labelsTs", "*.tif")))

    assert len(all_predictions) == len(all_gt)

    dice_scores = []
    for prediction_path, gt_path in tqdm(zip(all_predictions, all_gt), total=len(all_gt)):
        gt = imageio.imread(gt_path)
        prediction = imageio.imread(prediction_path)

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
    elif args.dataset in DATASET_MAPPING_3D:
        dmap_base = DATASET_MAPPING_3D
    else:
        raise ValueError(args.dataset)

    _, dataset_name = dmap_base[args.dataset]

    root_dir = os.path.join(NNUNET_ROOT, "test", dataset_name)
    evaluate_predictions(root_dir, args.dataset)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    args = parser.parse_args()
    main(args)
