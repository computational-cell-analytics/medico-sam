import os
import argparse
from glob import glob
from natsort import natsorted


VALID_DATASETS = [
    "sega", "uwaterloo_skin", "idrid", "camus", "montgomery",
]

FEXT = {
    "sega": ["kits", "rider", "dongyang"],
}

SEMANTIC_CLASS_MAPS = {
    "sega": {"aorta": 1},
}

ROOT = "/scratch/share/cidas/cca/data"


def get_dataset_paths(dataset_name, split="test"):
    assert dataset_name in VALID_DATASETS

    dirext = f"{dataset_name}/slices/"
    if dataset_name in FEXT:
        dirext += f"{FEXT[dataset_name]}/"
    dirext += f"{split}/"

    data_dir = os.path.join(ROOT, dirext)
    assert os.path.exists(data_dir), f"The data directory does not exist at '{data_dir}'."

    image_paths = glob(os.path.join(data_dir, "images", "*.tif"))
    gt_paths = glob(os.path.join(data_dir, "ground_truth", "*.tif"))

    return natsorted(image_paths), natsorted(gt_paths), SEMANTIC_CLASS_MAPS[data_dir]


def get_pred_paths(prediction_folder):
    pred_paths = sorted(glob(os.path.join(prediction_folder, "*")))
    return pred_paths


#
# PARSER FOR ALL THE REQUIRED ARGUMENTS
#


def get_default_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Provide the model type to initialize the predictor"
    )
    parser.add_argument("-c", "--checkpoint", type=none_or_str, required=True, default=None)
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("--box", action="store_true", help="If passed, starts with first prompt as box")
    parser.add_argument(
        "--use_masks", action="store_true", help="To use logits masks for iterative prompting."
    )
    args = parser.parse_args()
    return args


def none_or_str(value):
    if value == 'None':
        return None
    return value
