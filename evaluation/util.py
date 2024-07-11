import os
import shutil
import argparse
from glob import glob
from natsort import natsorted

import torch


VALID_DATASETS = [
    "sega", "uwaterloo_skin", "idrid", "camus", "montgomery", "oimhs", "btcv", "btcv_3d", "isic", "dca1",
    "papila", "m2caiseg", "siim_acr", "jnu-ifm", "cbis_ddsm", "piccolo", "duke_liver", "spider", "microusp",
    "han-seg", "toothfairy", "drive",
]

DEXT = {
    "sega": ["slices/kits", "slices/rider", "slices/dongyang"],
    "camus": ["slices/2ch", "slices/4ch"],
    "btcv_3d": ["slices_3d"],
    "papila": ["slices/cup", "slices/disc"],
}

SEMANTIC_CLASS_MAPS = {
    "sega": {"aorta": 1},
    "uwaterloo_skin": {"skin_lesion": 1},
    "idrid": {"optic_disc": 1},
    "montgomery": {"lungs": 1},
    "camus": {"endocardium": 1, "left_ventricle": 2, "left_atrium": 3},
    "oimhs": {"choroid": 1, "retina": 2, "intraretinal_cysts": 3, "macular_hole": 4},
    "btcv": {
        "background": 0, "spleen": 1, "right_kidney": 2, "left_kidney": 3, "gallbladder": 4, "esophagus": 5,
        "liver": 6, "stomach": 7, "aorta": 8, "inferior_vena_cava": 9, "portan_vein_and_splenic_vein": 10,
        "pancreas": 11, "right_adrenal_gland": 12, "left_adrenal_gland": 13,
    },
    "btcv_3d": {"aorta": 8},
    "isic": {"skin_lesion": 255},
    "dca1": {"vessel": 255},
    "papila": {"oc_or_od": 1},
    "osic_pulmofib": {"heart": 1, "lung": 2, "trachea": 3},
    "m2caiseg": {
        "grasper": 1, "bipolar": 2, "hook": 3, "scissors": 4, "clipper": 5, "irrigator": 6,
        "specimen_bag": 7, "trocars": 8, "clip": 9, "liver": 10, "gall_bladder": 11, "fat": 12,
        "upper_wall": 13, "artery": 14, "intestine": 15, "bile": 16, "blood": 17, "unknown": 18,
    },
    "siim_acr": {"pneumothorax": 255},
    "jnu-ifm": {"pubic_symphysis": 1, "fetal_head": 2},
    "cbis_ddsm": {"mass": 255},
    "piccolo": {"polyp": 255},
    "duke_liver": {"liver": 1},
    "microusp": {"prostate": 1},
    "spider": {"all": None},
    "han-seg": {
        "A_Carotid_L": 1, "A_Carotid_R": 2, "Arytenoid": 3, "Bone_Mandible": 4, "Brainstem": 5, "BuccalMucosa": 6,
        "Cavity_Oral": 7, "Cochlea_L": 8, "Cochlea_R": 9, "Cricopharyngeus": 10, "Esophagus_S": 11, "Eye_AL": 12,
        "Eye_AR": 13, "Eye_PL": 14, "Eye_PR": 15, "Glnd_Lacrimal_L": 16, "Glnd_Lacrimal_R": 17, "Glnd_Submand_L": 18,
        "Glnd_Submand_R": 19, "Glnd_Thyroid": 20, "Glottis": 21, "Larynx_SG": 22, "Lips": 23, "OpticChiasm": 24,
        "OpticNrv_L": 25, "OpticNrv_R": 26, "Parotid_L": 27, "Parotid_R": 28, "Pituitary": 29, "SpinalCord": 30,
    },
    "toothfairy": {"mandibular_canal": 1},
    "drive": {"vessel": 1}
}

MULTICLASS_SEMANTIC = ["oimhs", "btcv", "m2caiseg", "jnu-ifm", "osic_pulmofib", "spider", "han-seg"]

ROOT = "/scratch/share/cidas/cca/data"


def get_dataset_paths(dataset_name, split="test"):
    assert dataset_name in VALID_DATASETS

    if dataset_name in DEXT:
        dexts = DEXT[dataset_name]
    else:
        dexts = ["slices"]

    image_paths, gt_paths = [], []
    for per_dext in dexts:
        data_dir = os.path.join(ROOT, dataset_name.split("_3d")[0], per_dext, split)
        assert os.path.exists(data_dir), f"The data directory does not exist at '{data_dir}'."

        if dataset_name.endswith("_3d"):
            ext = "*.nii.gz"
        else:
            ext = "*tif"

        image_paths.extend(glob(os.path.join(data_dir, "images", ext)))
        gt_paths.extend(glob(os.path.join(data_dir, "ground_truth", ext)))

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
        "-m", "--model", type=str, default="vit_b", help="Provide the model type to initialize the predictor"
    )
    parser.add_argument("-c", "--checkpoint", type=none_or_str, required=True, default=None)
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("--box", action="store_true", help="If passed, starts with first prompt as box")
    parser.add_argument("--use_masks", action="store_true", help="To use logits masks for iterative prompting.")

    # for SAM-Med2d models
    parser.add_argument("--use_sam_med2d", action="store_true", help="Whether to use the SAM-Med2d model.")
    parser.add_argument("--adapter", action="store_true", help="Whether the model has the adapter blocks or not.")

    # use lora
    parser.add_argument("--lora_rank", default=None, type=int, help="The rank for LoRA.")
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


def _export_multi_gpu_models(checkpoint_path, save_path):
    from collections import OrderedDict
    from micro_sam.util import _load_checkpoint

    _, model_state = _load_checkpoint(checkpoint_path=checkpoint_path)

    sam_prefix = "module.sam."
    model_state = OrderedDict(
        [(k[len(sam_prefix):] if k.startswith(sam_prefix) else k, v) for k, v in model_state.items()]
    )

    torch.save(model_state, save_path)
    print(f"Model is saved at {save_path}.")


def _export_all_models():
    root_dir = "/scratch/share/cidas/cca/models/"

    # generalist
    _export_multi_gpu_models(
        checkpoint_path=os.path.join(
            root_dir, "medico-sam/multi_gpu/checkpoints/vit_b/medical_generalist_sam_multi_gpu/best.pt"
        ),
        save_path=os.path.join(
            root_dir, "medico-sam/multi_gpu/checkpoints/vit_b/medical_generalist_sam_multi_gpu/best_exported.pt"
        ),
    )

    # medsam
    _export_multi_gpu_models(
        checkpoint_path=os.path.join(
            root_dir, "medsam/multi_gpu/checkpoints/vit_b/medical_generalist_medsam_multi_gpu/best.pt"
        ),
        save_path=os.path.join(
            root_dir, "medsam/multi_gpu/checkpoints/vit_b/medical_generalist_medsam_multi_gpu/best_exported.pt"
        ),
    )

    # simplesam
    _export_multi_gpu_models(
        checkpoint_path=os.path.join(
            root_dir, "simplesam/multi_gpu/checkpoints/vit_b/medical_generalist_simplesam_multi_gpu/best.pt"
        ),
        save_path=os.path.join(
            root_dir, "simplesam/multi_gpu/checkpoints/vit_b/medical_generalist_simplesam_multi_gpu/best_exported.pt"
        ),
    )
