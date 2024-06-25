import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Union, List

import json
import numpy as np
import imageio.v3 as imageio

from common import _get_paths, DATA_ROOT, NNUNET_ROOT, create_json_files


ID_MAP = {
    r"1. Microaneurysms": 1,
    r"2. Haemorrhages": 2,
    r"3. Hard Exudates": 3,
    r"4. Soft Exudates": 4,
    r"5. Optic Disc": 5,
}


def _write_dataset_json_file(trg_dir, dataset_name):
    gt_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "labelsTr")

    train_ids = [Path(_path).stem for _path in glob(os.path.join(gt_dir, "*_train.tif"))]
    val_ids = [Path(_path).stem for _path in glob(os.path.join(gt_dir, "*_val.tif"))]

    json_file = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "dataset.json")

    data = {
        "channel_names": {
            "0": "Fundus"
        },
        "labels": {
            "background": 0,
            "microaneurysm": 1,
            "haemorrhages": 2,
            "hard_exudates": 3,
            "soft_exudates": 4,
            "optic_disc": 5,
        },
        "numTraining": len(val_ids) + len(train_ids),
        "file_ending": ".tif",
        "name": dataset_name,
        "description": "IDRiD: https://doi.org/10.1016/j.media.2019.101561"
    }

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    return train_ids, val_ids


def _update_label_image(shape, input_paths: List[Union[os.PathLike, str]], trg_path: Union[os.PathLike, str]):
    semantic_gt = np.zeros(shape)
    for input_path in input_paths:
        cname = input_path.split("/")[-2]
        per_class_gt = imageio.imread(input_path)
        semantic_gt[per_class_gt] = ID_MAP[cname]

    imageio.imwrite(trg_path, semantic_gt)


def _update_input_image(image_path, trg_image_path):
    image = imageio.imread(image_path)
    imageio.imwrite(trg_image_path, image)


def convert_idrid_for_training(path, trg_dir, dataset_name):
    train_image_paths, train_gt_paths = _get_paths(path, "idrid", "train")
    val_image_paths, val_gt_paths = _get_paths(path, "idrid", "val")

    # the idea is we move all the images to one directory, write their image ids into a split.json file,
    # which nnunet will read to define the custom validation split
    image_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "imagesTr")
    gt_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "labelsTr")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    def _move_per_split(split, image_paths, gt_paths):
        _ids = []
        for image_path in tqdm(sorted(image_paths)):
            image_id = Path(image_path).stem

            trg_image_path = os.path.join(image_dir, f"{image_id}_{split}_0000.tif")
            _update_input_image(image_path, trg_image_path)

            per_image_gt_paths = [_path for _path in gt_paths if image_id in _path]

            trg_gt_path = os.path.join(gt_dir, f"{image_id}_{split}.tif")
            _update_label_image(
                shape=imageio.imread(image_path).shape[:2], input_paths=per_image_gt_paths, trg_path=trg_gt_path
            )

            _ids.append(Path(trg_gt_path).stem)

        return _ids

    _move_per_split("train", train_image_paths, train_gt_paths)
    _move_per_split("val", val_image_paths, val_gt_paths)


def main():
    path = os.path.join(DATA_ROOT, "idrid")
    dataset_name = "Dataset202_IDRiD"

    # space to store your top-level nnUNet files
    trg_root = NNUNET_ROOT

    convert_idrid_for_training(path=path, trg_dir=trg_root, dataset_name=dataset_name)
    create_json_files(trg_dir=trg_root, dataset_name=dataset_name, write_json_function=_write_dataset_json_file)


if __name__ == "__main__":
    main()
