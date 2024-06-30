import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path

import json

from common import _get_paths, DATA_ROOT, NNUNET_ROOT, create_json_files


def _write_dataset_json_file(trg_dir, dataset_name):
    gt_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "labelsTr")

    train_ids = [Path(_path).stem.split(".")[0] for _path in glob(os.path.join(gt_dir, "*_train.nii.gz"))]
    val_ids = [Path(_path).stem.split(".")[0] for _path in glob(os.path.join(gt_dir, "*_val.nii.gz"))]

    json_file = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "dataset.json")

    data = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": {
            "background": 0,
            "liver": 1,
        },
        "numTraining": len(val_ids) + len(train_ids),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "Duke Liver: https://zenodo.org/records/7774566"
    }

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    return train_ids, val_ids


def convert_duke_liver_for_training(path, trg_dir, dataset_name):
    train_image_paths, train_gt_paths = _get_paths(path, "duke_liver", "train")
    val_image_paths, val_gt_paths = _get_paths(path, "duke_liver", "val")

    # the idea is we move all the images to one directory, write their image ids into a split.json file,
    # which nnunet will read to define the custom validation split
    image_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "imagesTr")
    gt_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "labelsTr")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    def _move_per_split(split, image_paths, gt_paths):
        _ids = []
        assert len(gt_paths) == len(image_paths)
        for image_path, gt_path in tqdm(
            zip(sorted(image_paths), sorted(gt_paths)), total=len(image_paths)
        ):
            image_id = Path(image_path).stem
            image_id = image_id.split(".")[0]

            trg_image_path = os.path.join(image_dir, f"{image_id}_{split}_0000.nii.gz")
            shutil.copy(src=image_path, dst=trg_image_path)

            trg_gt_path = os.path.join(gt_dir, f"{image_id}_{split}.nii.gz")
            shutil.copy(src=gt_path, dst=trg_gt_path)

            _ids.append(Path(trg_gt_path).stem)

        return _ids

    _move_per_split("train", train_image_paths, train_gt_paths)
    _move_per_split("val", val_image_paths, val_gt_paths)


def main():
    path = os.path.join(DATA_ROOT, "duke_liver")
    dataset_name = "Dataset305_DukeLiver"

    # space to store your top-level nnUNet files
    trg_root = NNUNET_ROOT

    convert_duke_liver_for_training(path=path, trg_dir=trg_root, dataset_name=dataset_name)
    create_json_files(trg_dir=trg_root, dataset_name=dataset_name, write_json_function=_write_dataset_json_file)


if __name__ == "__main__":
    main()
