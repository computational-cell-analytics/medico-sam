import os
import shutil
import argparse
from glob import glob
from natsort import natsorted

import torch


VALID_DATASETS = [
    "sega", "uwaterloo_skin", "idrid", "camus", "montgomery", "oimhs"
]

DEXT = {
    "sega": ["slices/kits", "slices/rider", "slices/dongyang"],
    "camus": ["slices/2ch", "slices/4ch"]
}

SEMANTIC_CLASS_MAPS = {
    "sega": {"aorta": 1},
    "uwaterloo_skin": {"skin_lesion": 1},
    "idrid": {"optic_disc": 1},
    "montgomery": {"lungs": 1},
    "camus": {"A": 1, "B": 2, "C": 3},
    "oimhs": {"choroid": 1, "retina": 2, "intraretinal_cysts": 3, "macular_hole": 4}
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
    pred_paths = natsorted(glob(os.path.join(prediction_folder, "*")))
    return pred_paths


def _clear_files(experiment_folder, semantic_class_maps):
    # Check if both the results from iterative prompting starting box and points are there.
    _completed_inference = []
    for cname in semantic_class_maps.keys():
        res_dir = os.path.join(experiment_folder, "results", cname)
        box_rpath = os.path.join(res_dir, "iterative_prompts_start_box.csv")
        point_rpath = os.path.join(res_dir, "iterative_prompts_start_point.csv")
        semantic_rpath = os.path.join(res_dir, "semantic_segmentation.csv")

        if os.path.exists(box_rpath) and os.path.exists(point_rpath) and os.path.exists(semantic_rpath):
            _completed_inference.append(True)
        else:
            _completed_inference.append(False)

    if len(_completed_inference) > 0 and all(_completed_inference):
        if os.path.exists(os.path.join(experiment_folder, "embeddings")):
            shutil.rmtree(os.path.join(experiment_folder, "embeddings"))
        shutil.rmtree(os.path.join(experiment_folder, "start_with_point"))
        shutil.rmtree(os.path.join(experiment_folder, "start_with_box"))
        shutil.rmtree(os.path.join(experiment_folder, "semantic_segmentation"))


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
    parser.add_argument("--use_masks", action="store_true", help="To use logits masks for iterative prompting.")

    # for SAM-Med2d models
    parser.add_argument("--use_sam_med2d", action="store_true", help="Whether to use the SAM-Med2d model.")
    parser.add_argument("--adapter", action="store_true", help="Whether the model has the adapter blocks or not.")
    args = parser.parse_args()

    if args.adapter:
        assert args.use_sam_med2d, "You need to use SAM-Med2d model as adapters are only supported for SAM-Med2d atm."

    return args


def none_or_str(value):
    if value == 'None':
        return None
    return value


#
# EXPERIMENTAL SCRIPTS
#


def _convert_sam_med2d_models(checkpoint_path, save_path):
    if os.path.exists(save_path):
        return

    state = torch.load(checkpoint_path)
    model_state = state["model"]
    torch.save(model_state, save_path)


def test_medical_sam_models():
    # COMPATIBLE
    # ckpt_path = "/scratch-grete/projects/nim00007/sam/models/medsam/medsam_vit_b.pth"
    # save_path = None

    # 1. STATE NEEDS TO BE UPDATED
    # 2. It changes the input patch shape, hence the SAM model needs to be adapted likewise (inconvenient)
    # ckpt_path = "/scratch-grete/projects/nim00007/sam/models/sam-med2d/ft-sam_b.pth"
    # adapter = False
    # save_path = "/scratch-grete/projects/nim00007/sam/models/sam-med2d/ft-sam_b_model.pt"

    # 1. STATE NEEDS TO BE UPDATED
    # 2. It changes the input patch shape, hence the SAM model needs to be adapted likewise (inconvenient)
    # 3. ADAPTER BLOCKS NEED TO BE ADDED
    ckpt_path = "/scratch-grete/projects/nim00007/sam/models/sam-med2d/sam-med2d_b.pth"
    adapter = True
    # save_path = "/scratch-grete/projects/nim00007/sam/models/sam-med2d/sam-med2d_b_model.pt"

    # if save_path is not None:
    #     _convert_sam_med2d_models(checkpoint_path=ckpt_path, save_path=save_path)

    # from micro_sam.util import get_sam_model
    # _ = get_sam_model(model_type="vit_b", checkpoint_path=ckpt_path if save_path is None else save_path)

    from medico_sam.model.util import get_sam_med2d_model
    _ = get_sam_med2d_model(model_type="vit_b", checkpoint_path=ckpt_path, encoder_adapter=adapter)

    print("Loading the model was successful.")
