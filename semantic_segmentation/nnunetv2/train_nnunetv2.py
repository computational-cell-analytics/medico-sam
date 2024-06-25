import os
import warnings
import argparse

import torch


NNUNET_ROOT = "/scratch/share/cidas/cca/nnUNetv2"

DATASET_MAPPING = {
    "oimhs": [201, "Dataset201_OIMHS"],
    "idrid": [202, "Dataset202_IDRiD"],
    "isic": [203, "Dataset203_ISIC"],
    "dca1": [204, "Dataset204_DCA1"],
}


def declare_paths(nnunet_path: str):
    """To let the system known of the path variables where the respective folders exist (important for all components)
    """
    warnings.warn(
        "Make sure you have created the directories mentioned in this functions (relative to the root directory)"
    )

    os.environ["nnUNet_raw"] = os.path.join(nnunet_path, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(nnunet_path, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(nnunet_path, "nnUNet_results")


def preprocess_data(dataset_id):
    # let's check the preprocessing first
    cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
    os.system(cmd)


def train_nnunetv2(fold, dataset_name, dataset_id):
    _have_splits = os.path.exists(
        os.path.join(NNUNET_ROOT, "nnUNet_preprocessed", dataset_name, "splits_final.json")
    )
    assert _have_splits, "The experiment expects you to create the splits yourself."

    # train 2d nnUNet
    gpus = torch.cuda.device_count()
    cmd = f"nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train {dataset_id} 2d {fold} -num_gpus {gpus} --c"
    os.system(cmd)


def predict_nnunetv2(fold, dataset_name, dataset_id):
    input_dir = os.path.join(NNUNET_ROOT, "test", dataset_name, "imagesTs")
    assert os.path.exists(input_dir)

    output_dir = os.path.join(NNUNET_ROOT, "test", dataset_name, "predictionTs")

    cmd = f"nnUNetv2_predict -i {input_dir} -o {output_dir} -d {dataset_id} -c 2d -f {fold}"
    os.system(cmd)


def main(args):
    declare_paths(NNUNET_ROOT)

    dataset_id, dataset_name = DATASET_MAPPING[args.dataset]

    if args.preprocess:
        preprocess_data(dataset_id=dataset_id)

    if args.train:
        train_nnunetv2(fold=args.fold, dataset_name=dataset_name, dataset_id=dataset_id)

    if args.predict:
        predict_nnunetv2(fold=args.fold, dataset_name=dataset_name, dataset_id=dataset_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)

    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    parser.add_argument("--fold", type=str, default="0")

    args = parser.parse_args()
    main(args)
