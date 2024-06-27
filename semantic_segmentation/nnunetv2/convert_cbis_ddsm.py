import os
from glob import glob
from tqdm import tqdm
from pathlib import Path

import json
import imageio.v3 as imageio

from common import _get_paths, DATA_ROOT, NNUNET_ROOT, create_json_files


def _write_dataset_json_file(trg_dir, dataset_name):
    gt_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "labelsTr")

    train_ids = [Path(_path).stem for _path in glob(os.path.join(gt_dir, "*_train.tif"))]
    val_ids = [Path(_path).stem for _path in glob(os.path.join(gt_dir, "*_val.tif"))]

    json_file = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "dataset.json")

    data = {
        "channel_names": {
            "0": "Mammography",
        },
        "labels": {
            "background": 0,
            "mass": 1,
        },
        "numTraining": len(val_ids) + len(train_ids),
        "file_ending": ".tif",
        "name": dataset_name,
        "description": "CBIS-DDSM: https://doi.org/10.1038/sdata.2017.177"
    }

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    return train_ids, val_ids


def convert_cbis_ddsm_for_training(path, trg_dir, dataset_name):
    train_image_paths, train_gt_paths = _get_paths(path, "cbis_ddsm", "train")
    val_image_paths, val_gt_paths = _get_paths(path, "cbis_ddsm", "val")

    # the idea is we move all the images to one directory, write their image ids into a split.json file,
    # which nnunet will read to define the custom validation split
    image_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "imagesTr")
    gt_dir = os.path.join(trg_dir, "nnUNet_raw", dataset_name, "labelsTr")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    def _move_per_split(split, image_paths, gt_paths):
        _ids = []
        counter = 0
        assert len(gt_paths) == len(image_paths)
        for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
            image_id = Path(image_path).stem

            if imageio.imread(image_path).shape != imageio.imread(gt_path).shape:
                print(f"Found mismatch number {counter}.")
                counter += 1
                continue

            trg_image_path = os.path.join(image_dir, f"{image_id}_{split}_0000.tif")
            imageio.imwrite(trg_image_path, imageio.imread(image_path))

            trg_gt_path = os.path.join(gt_dir, f"{image_id}_{split}.tif")
            imageio.imwrite(trg_gt_path, (imageio.imread(gt_path) > 1).astype("uint8"))

            _ids.append(Path(trg_gt_path).stem)

        return _ids

    _move_per_split("train", train_image_paths, train_gt_paths)
    _move_per_split("val", val_image_paths, val_gt_paths)


def convert_cbis_ddsm_for_testing(path, trg_dir, dataset_name):
    test_image_paths, test_gt_paths = _get_paths(path, "cbis_ddsm", "test")

    # the idea for here is to move the data to a central location,
    # where we can automate the inference procedure
    image_dir = os.path.join(trg_dir, "test", dataset_name, "imagesTs")
    gt_dir = os.path.join(trg_dir, "test", dataset_name, "labelsTs")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    counter = 0
    assert len(test_image_paths) == len(test_gt_paths)
    for image_path, gt_path in tqdm(zip(test_image_paths, test_gt_paths), total=len(test_image_paths)):
        image_id = Path(image_path).stem

        if imageio.imread(image_path).shape != imageio.imread(gt_path).shape:
            print(f"Found mismatch number {counter}.")
            counter += 1
            continue

        trg_image_path = os.path.join(image_dir, f"{image_id}_0000.tif")
        imageio.imwrite(trg_image_path, imageio.imread(image_path))

        trg_gt_path = os.path.join(gt_dir, f"{image_id}.tif")
        imageio.imwrite(trg_gt_path, (imageio.imread(gt_path) > 1).astype("uint8"))


def main():
    path = os.path.join(DATA_ROOT, "cbis_ddsm")
    dataset_name = "Dataset206_CBISDDSM"

    # space to store your top-level nnUNet files
    trg_root = NNUNET_ROOT

    convert_cbis_ddsm_for_training(path=path, trg_dir=trg_root, dataset_name=dataset_name)
    create_json_files(trg_dir=trg_root, dataset_name=dataset_name, write_json_function=_write_dataset_json_file)

    convert_cbis_ddsm_for_testing(path=path, trg_dir=trg_root, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
