import os
import argparse

from tukra.training import nnunet
from tukra.utils import nnunet_utils

from _common import _get_paths, _get_per_dataset_items, DATA_ROOT


NNUNET_ROOT = "/mnt/vast-nhr/projects/cidas/cca/nnUNetv2"

DATASET_MAPPING_2D = {
    "oimhs": [201, "Dataset201_OIMHS"],
    "isic": [202, "Dataset202_ISIC"],
    "dca1": [203, "Dataset203_DCA1"],
    "cbis_ddsm": [204, "Dataset204_CBISDDSM"],
    "drive": [205, "Dataset205_DRIVE"],
    "piccolo": [206, "Dataset206_PICCOLO"],
    "siim_acr": [207, "Dataset207_SIIM_ACR"],
    "hil_toothseg": [208, "Dataset208_HIL_ToothSeg"],
    "covid_qu_ex": [209, "Dataset209_Covid_QU_EX"],
}

DATASET_MAPPING_3D = {
    "curvas": [301, "Dataset301_Curvas"],
    "osic_pulmofib": [302, "Dataset302_OSICPulmoFib"],
    "sega": [303, "Dataset303_SegA"],
    "duke_liver": [304, "Dataset304_DukeLiver"],
    "oasis": [305, "Dataset305_OASIS"],
    "lgg_mri": [306, "Dataset306_LGG_MRI"],
    "leg_3d_us": [307, "Dataset307_Leg_3D_US"],
    "micro_usp": [308, "Dataset308_MicroUSP"],
}


def main(args):
    nnunet.declare_paths(NNUNET_ROOT)

    if args.dataset in DATASET_MAPPING_2D:
        dmap_base = DATASET_MAPPING_2D
        dim = "2d"
    elif args.dataset in DATASET_MAPPING_3D:
        dmap_base = DATASET_MAPPING_3D
        dim = "3d_fullres"
    else:
        raise ValueError(f"{args.dataset} is not a supported dataset.")

    dataset_id, nnunet_dataset_name = dmap_base[args.dataset]

    def _get_paths_per_dataset(split):
        image_paths, label_paths = _get_paths(
            path=os.path.join(DATA_ROOT, args.dataset), dataset=args.dataset, split=split
        )
        return image_paths, label_paths

    if args.preprocess:
        train_image_paths, train_label_paths = _get_paths_per_dataset(split="train")
        val_image_paths, val_label_paths = _get_paths_per_dataset(split="val")

        (
            file_suffix, transfer_mode, dataset_json_template, preprocess_inputs, preprocess_labels, keys
        ) = _get_per_dataset_items(
            dataset=args.dataset,
            nnunet_dataset_name=nnunet_dataset_name,
            train_id_count=len(train_image_paths),
            val_id_count=len(val_image_paths),
        )

        kwargs = {
            "dataset_name": nnunet_dataset_name,
            "file_suffix": file_suffix,
            "transfer_mode": transfer_mode,
            "preprocess_inputs": preprocess_inputs,
            "preprocess_labels": preprocess_labels,
            "ensure_unique": True if args.dataset in ["curvas", "leg_3d_us", "oasis"] else False,
            "keys": keys,
        }

        train_ids = nnunet_utils.convert_dataset_for_nnunet_training(
            image_paths=train_image_paths, gt_paths=train_label_paths, split="train", **kwargs
        )
        val_ids = nnunet_utils.convert_dataset_for_nnunet_training(
            image_paths=val_image_paths, gt_paths=val_label_paths, split="val", **kwargs
        )

        nnunet_utils.create_json_files(
            dataset_name=nnunet_dataset_name,
            file_suffix=file_suffix,
            dataset_json_template=dataset_json_template,
            train_ids=train_ids,
            val_ids=val_ids,
        )

        nnunet.preprocess_data(dataset_id=dataset_id)

    if args.train:
        # TODO: train other folds
        nnunet.train_nnunetv2(fold=args.fold, dataset_name=nnunet_dataset_name, dataset_id=dataset_id, dim=dim)

    if args.predict:
        test_image_paths, test_label_paths = _get_paths_per_dataset(split="test")
        file_suffix, transfer_mode, _, preprocess_inputs, preprocess_labels, keys = _get_per_dataset_items(
            dataset=args.dataset,
            nnunet_dataset_name=nnunet_dataset_name,
            train_id_count=None,
            val_id_count=None,
        )

        kwargs = {
            "dataset_name": nnunet_dataset_name,
            "file_suffix": file_suffix,
            "transfer_mode": transfer_mode,
            "preprocess_inputs": preprocess_inputs,
            "preprocess_labels": preprocess_labels,
            "ensure_unique": True if args.dataset in ["curvas", "leg_3d_us", "oasis"] else False,
            "keys": keys,
        }

        nnunet_utils.convert_dataset_for_nnunet_training(
            image_paths=test_image_paths, gt_paths=test_label_paths, split="test", **kwargs
        )

        nnunet.predict_nnunetv2(fold=args.fold, dataset_name=nnunet_dataset_name, dataset_id=dataset_id, dim=dim)

    print("The process has finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)

    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    parser.add_argument("--fold", type=str, default="0")

    args = parser.parse_args()
    main(args)
