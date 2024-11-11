import argparse

from tukra.training import nnunet
from tukra.utils import nnunet_utils

from common import _get_paths, _get_per_dataset_items, DATA_ROOT


NNUNET_ROOT = "/scratch/share/cidas/cca/nnUNetv2"

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
    "toothfairy": [305, "Dataset305_ToothFairy"],
    "oasis": [306, "Dataset306_OASIS"],
    "lgg_mri": [307, "Dataset307_LGG_MRI"],
    "leg_3d_us": [308, "Dataset308_Leg_3D_US"],
    "micro_usp": [309, "Dataset309_MicroUSP"],
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
        raise ValueError(args.dataset)

    dataset_id, dataset_name = dmap_base[args.dataset]

    if args.preprocess:
        (
            file_suffix, transfer_mode, dataset_json_template, preprocess_inputs, preprocess_labels
        ) = _get_per_dataset_items(dataset=args.dataset)

        kwargs = {
            "dataset_name": dataset_name,
            "file_suffix": file_suffix,
            "transfer_mode": transfer_mode,
            "preprocess_inputs": preprocess_inputs,
            "preprocess_labels": preprocess_labels,
        }

        train_image_paths, train_label_paths = _get_paths(path=DATA_ROOT, dataset=args.dataset, split="train")
        train_ids = nnunet_utils.convert_dataset_for_nnunet_training(
            image_paths=train_image_paths, gt_paths=train_label_paths, split="train", **kwargs
        )

        val_image_paths, val_label_paths = _get_paths(path=DATA_ROOT, split="train")
        val_ids = nnunet_utils.convert_dataset_for_nnunet_training(
            image_paths=val_image_paths, gt_paths=val_label_paths, split="val", **kwargs
        )

        nnunet_utils.create_json_files(
            dataset_name=dataset_name,
            file_suffix=file_suffix,
            dataset_json_template=dataset_json_template,
            train_ids=train_ids,
            val_ids=val_ids,
        )

        nnunet.preprocess_data(dataset_id=dataset_id)

    if args.train:
        nnunet.train_nnunetv2(fold=args.fold, dataset_name=dataset_name, dataset_id=dataset_id, dim=dim)

    if args.predict:
        test_image_paths, test_label_paths = _get_paths(path=DATA_ROOT, dataset=args.dataset, split="test")
        nnunet_utils.convert_dataset_for_nnunet_training(
            image_paths=test_image_paths, gt_paths=test_label_paths, split="test", **kwargs
        )

        nnunet.predict_nnunetv2(fold=args.fold, dataset_name=dataset_name, dataset_id=dataset_id, dim=dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)

    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    parser.add_argument("--fold", type=str, default="0")

    args = parser.parse_args()
    main(args)
