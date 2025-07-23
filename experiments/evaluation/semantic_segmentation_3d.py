import os
from glob import glob
from natsort import natsorted

import torch

from torch_em.util import load_model

from medico_sam.evaluation import inference
from medico_sam.util import get_semantic_sam_model
from medico_sam.evaluation.evaluation import run_evaluation_for_semantic_segmentation

from util import get_default_arguments


DATASET_MAPPING_3D = {
    "osic_pulmofib": "Dataset302_OSICPulmoFib",
    "duke_liver": "Dataset304_DukeLiver",
    "oasis": "Dataset305_OASIS",
    "lgg_mri": "Dataset306_LGG_MRI",
    "leg_3d_us": "Dataset307_Leg_3D_US",
    "micro_usp": "Dataset308_MicroUSP",
}

CLASS_MAPS = {
    # 3d datasets
    "osic_pulmofib": {"heart": 1, "lung": 2, "trachea": 3},
    "duke_liver": {"liver": 1},
    "oasis": {"gray_matter": 1, "thalamus": 2, "white_matter": 3, "csf": 4},
    "lgg_mri": {"glioma": 1},
    "leg_3d_us": {"SOL": 1, "GM": 2, "GL": 3},
    "micro_usp": {"prostate": 1},
}


def get_3d_dataset_paths(dataset_name):
    root_dir = os.path.join("/mnt/vast-nhr/projects/cidas/cca/nnUNetv2/nnUNet_raw", DATASET_MAPPING_3D[dataset_name])
    image_paths = natsorted(glob(os.path.join(root_dir, "imagesTs", "*")))
    gt_paths = natsorted(glob(os.path.join(root_dir, "labelsTs", "*")))
    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0
    return image_paths, gt_paths, CLASS_MAPS[dataset_name]


def main():
    args = get_default_arguments()

    assert args.dataset is not None

    image_paths, gt_paths, semantic_class_maps = get_3d_dataset_paths(dataset_name=args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_semantic_sam_model(
        model_type="vit_b",
        num_classes=len(semantic_class_maps) + 1,
        ndim=3,
        peft_kwargs=None if args.lora_rank is None else {"rank": args.lora_rank},
        device=device,
    )
    model = load_model(args.checkpoint, device=device, model=model)
    model.to(device)
    model.eval()

    # Whether to make channels first or not.
    make_channels_first = False if args.dataset in ["lgg_mri", "leg_3d_us", "oasis"] else True

    inference.run_semantic_segmentation_3d(
        model=model,
        image_paths=image_paths,
        prediction_dir=args.experiment_folder,
        semantic_class_map=semantic_class_maps,
        is_multiclass=True,
        image_key=None,
        make_channels_first=make_channels_first,
    )

    run_evaluation_for_semantic_segmentation(
        gt_paths=gt_paths,
        prediction_root=args.experiment_folder,
        experiment_folder=args.experiment_folder,
        semantic_class_map=semantic_class_maps,
        is_multiclass=True,
        ensure_channels_first=make_channels_first,
    )


if __name__ == "__main__":
    main()
