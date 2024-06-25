import os

import json

from torch_em.data.datasets import medical


DATA_ROOT = "/scratch/share/cidas/cca/data"
NNUNET_ROOT = "/scratch/share/cidas/cca/nnUNetv2"


def _get_paths(path, dataset, split):
    if dataset == "oimhs":
        image_paths, gt_paths = medical.oimhs._get_oimhs_paths(path=path, split=split, download=False)

    elif dataset == "idrid":
        if split != "test":
            split = "train"

        image_paths, gt_paths = medical.idrid._get_idrid_paths(
            path=path, split=split, task="microaneurysms", download=False
        )
        gt_paths.extend(
            medical.idrid._get_idrid_paths(path=path, split=split, task="haemorrhages", download=False)[-1]
        )
        gt_paths.extend(
            medical.idrid._get_idrid_paths(path=path, split=split, task="hard_exudates", download=False)[-1]
        )
        gt_paths.extend(
            medical.idrid._get_idrid_paths(path=path, split=split, task="soft_exudates", download=False)[-1]
        )
        gt_paths.extend(
            medical.idrid._get_idrid_paths(path=path, split=split, task="optic_disc", download=False)[-1]
        )

        if split == "train":
            image_paths = image_paths[:-6]
        elif split == "val":
            image_paths = image_paths[-6:]

    elif dataset == "isic":
        image_paths, gt_paths = medical.isic._get_isic_paths(path=path, split=split, download=False)

    elif dataset == "dca1":
        image_paths, gt_paths = medical.dca1._get_dca1_paths(path=path, split=split, download=False)

    else:
        raise ValueError(dataset)

    return image_paths, gt_paths


def create_json_files(trg_dir, dataset_name, write_json_function):
    # now, let's create the 'dataset.json' file based on the available inputs
    train_ids, val_ids = write_json_function(trg_dir=trg_dir, dataset_name=dataset_name)

    # let's try to store the splits file already
    preprocessed_dir = os.path.join(trg_dir, "nnUNet_preprocessed", dataset_name)
    os.makedirs(preprocessed_dir, exist_ok=True)

    # we create custom splits for all folds, to fit with the expectation
    all_split_inputs = [{'train': train_ids, 'val': val_ids} for _ in range(5)]
    json_file = os.path.join(preprocessed_dir, "splits_final.json")
    with open(json_file, "w") as f:
        json.dump(all_split_inputs, f, indent=4)
