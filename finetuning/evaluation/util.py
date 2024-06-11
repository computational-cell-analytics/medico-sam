import os
import shutil
import argparse
from glob import glob
from natsort import natsorted


VALID_DATASETS = [
    "sega", "uwaterloo_skin", "idrid", "camus", "montgomery",
]

DEXT = {
    "sega": ["slices/kits", "slices/rider", "slices/dongyang"],
}

SEMANTIC_CLASS_MAPS = {
    "sega": {"aorta": 1},
}

ROOT = "/scratch/share/cidas/cca/data"


def get_dataset_paths(dataset_name, split="test"):
    assert dataset_name in VALID_DATASETS

    if dataset_name in DEXT:
        dexts = DEXT[dataset_name]
    else:
        dexts = ["slices"]

    image_paths, gt_paths = [], []
    for per_dext in dexts:
        data_dir = os.path.join(ROOT, dataset_name, per_dext, split)
        assert os.path.exists(data_dir), f"The data directory does not exist at '{data_dir}'."

        image_paths.extend(glob(os.path.join(data_dir, "images", "*.tif")))
        gt_paths.extend(glob(os.path.join(data_dir, "ground_truth", "*.tif")))

    assert len(image_paths) == len(gt_paths)

    return natsorted(image_paths), natsorted(gt_paths), SEMANTIC_CLASS_MAPS[dataset_name]


def get_pred_paths(prediction_folder):
    pred_paths = sorted(glob(os.path.join(prediction_folder, "*")))
    return pred_paths


def _clear_files(experiment_folder, semantic_class_maps):
    # Check if both the results from iterative prompting starting box and points are there.
    _completed_inference = []
    for cname in semantic_class_maps.keys():
        box_rpath = os.path.join(experiment_folder, "results", cname, "iterative_prompts_start_box.csv")
        point_rpath = os.path.join(experiment_folder, "results", cname, "iterative_prompts_start_point.csv")

        if os.path.exists(box_rpath) and os.path.exists(point_rpath):
            _completed_inference.append(True)
        else:
            _completed_inference.append(False)

    if all(_completed_inference) and len(_completed_inference) > 0:
        shutil.rmtree(os.path.join(experiment_folder, "embeddings"))
        shutil.rmtree(os.path.join(experiment_folder, "start_with_point"))
        shutil.rmtree(os.path.join(experiment_folder, "start_with_box"))


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
