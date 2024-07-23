import os
from glob import glob
from natsort import natsorted

from torch_em.util import load_model

from medico_sam.evaluation import inference
from medico_sam.util import get_medico_sam_model
from medico_sam.evaluation.evaluation import run_evaluation_for_semantic_segmentation

from util import get_default_arguments


DATASET_MAPPING_3D = {
    "btcv": "Dataset301_BTCV",
    "osic_pulmofib": "Dataset303_OSICPulmoFib",
    "sega": "Dataset304_SegA",
    "duke_liver": "Dataset305_DukeLiver",
}


def check_lucchi():
    ckpt = "/home/nimcpape/Work/my_projects/medico-sam/semantic_segmentation/checkpoints/lucchi_3d_adapter_lora4"
    model = get_medico_sam_model("vit_b", device="cuda", use_sam3d=True, lora_rank=4, n_classes=2, image_size=512)
    model = load_model(ckpt, device="cuda", model=model)

    input_paths = [
        "/home/nimcpape/Work/my_projects/medico-sam/semantic_segmentation/data/lucchi_test.h5"
    ]
    output_dir = "./pred_lucchi"
    inference.run_semantic_segmentation_3d(
        model, input_paths, output_dir, semantic_class_map={"blub": 0}, is_multiclass=True,
        image_key="raw",
    )


def get_3d_dataset_paths(dataset_name):
    root_dir = "/scratch/share/cidas/cca/nnUNetv2"
    image_paths = natsorted(glob(os.path.join(root_dir, "test", DATASET_MAPPING_3D[dataset_name], "imagesTs", "*")))
    gt_paths = natsorted(glob(os.path.join(root_dir, "test", DATASET_MAPPING_3D[dataset_name], "labelsTs", "*")))

    if dataset_name == "sega":
        semantic_maps = {"aorta": 1}
    elif dataset_name == "duke_liver":
        semantic_maps = {"liver": 1}
    elif dataset_name == "osic_pulmofib":
        semantic_maps = {"heart": 1, "lung": 2, "trachea": 3}
    elif dataset_name == "btcv":
        semantic_maps = {
            "spleen": 1, "right_kidney": 2, "left_kidney": 3, "gallbladder": 4, "esophagus": 5, "liver": 6,
            "stomach": 7, "aorta": 8, "inferior_vena_cava": 9, "portan_vein_and_splenic_vein": 10,
            "pancreas": 11, "right_adrenal_gland": 12, "left_adrenal_gland": 13,
        }

    assert len(image_paths) == len(gt_paths)

    return image_paths, gt_paths, semantic_maps


def main():
    args = get_default_arguments()

    assert args.dataset is not None

    image_paths, gt_paths, semantic_class_maps = get_3d_dataset_paths(dataset_name=args.dataset)

    model = get_medico_sam_model(
        model_type="vit_b",
        device="cuda",
        use_sam3d=True,
        lora_rank=4,
        n_classes=len(semantic_class_maps)+1,
        image_size=512
    )
    model = load_model(args.checkpoint, device="cuda", model=model)

    inference.run_semantic_segmentation_3d(
        model=model,
        image_paths=image_paths,
        prediction_dir=args.experiment_folder,
        semantic_class_map=semantic_class_maps,
        is_multiclass=True,
        image_key="raw",
        make_channels_first=True,
    )

    run_evaluation_for_semantic_segmentation(
        gt_paths=gt_paths,
        prediction_root=args.experiment_folder,
        experiment_folder=args.experiment_folder,
        semantic_class_map=semantic_class_maps,
        is_multiclass=True,
        for_3d=True,
    )


if __name__ == "__main__":
    main()
