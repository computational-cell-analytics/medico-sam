import os

import json

from torch_em.data.datasets import medical


DATA_ROOT = "/scratch/share/cidas/cca/data"
NNUNET_ROOT = "/scratch/share/cidas/cca/nnUNetv2"


def _get_paths(path, dataset, split):
    if dataset == "oimhs":
        image_paths, gt_paths = medical.oimhs.get_oimhs_paths(path=path, split=split, download=False)

    elif dataset == "cbis_ddsm":
        if split == "val":
            chosen_split = "Train"
        else:
            chosen_split = split.title()

        image_paths, gt_paths = medical.cbis_ddsm.get_cbis_ddsm_paths(
            path=path, split=chosen_split, task="Mass", tumour_type=None, download=True
        )

        if split == "train":
            image_paths, gt_paths = image_paths[125:], gt_paths[125:]
        elif split == "val":
            image_paths, gt_paths = image_paths[:125], gt_paths[:125]

    elif dataset == "isic":
        image_paths, gt_paths = medical.isic.get_isic_paths(path=path, split=split, download=False)

    elif dataset == "dca1":
        image_paths, gt_paths = medical.dca1.get_dca1_paths(path=path, split=split, download=False)

    elif dataset == "drive":
        image_paths, gt_paths = medical.drive.get_drive_paths(path=path, split=split, download=False)

    elif dataset == "piccolo":
        if split == "val":
            split = "validation"
        image_paths, gt_paths = medical.piccolo.get_piccolo_paths(path=path, split=split, download=False)

    elif dataset == "btcv":
        image_paths, gt_paths = medical.btcv._get_raw_and_label_paths(path=path, anatomy=["Abdomen"])
        image_paths, gt_paths = image_paths["Abdomen"], gt_paths["Abdomen"]

        if split == "train":
            image_paths, gt_paths = image_paths[:18], gt_paths[:18]
        elif split == "val":
            image_paths, gt_paths = image_paths[18:22], gt_paths[18:22]
        elif split == "test":
            image_paths, gt_paths = image_paths[22:], gt_paths[22:]

    elif dataset == "amos":
        image_paths, gt_paths = medical.amos.get_amos_paths(path=path, split=split, modality="MRI", download=False)

    elif dataset == "osic_pulmofib":
        image_paths, gt_paths = medical.osic_pulmofib.get_osic_pulmofib_paths(path=path, download=False)

        if split == "train":
            image_paths, gt_paths = image_paths[:75], gt_paths[:75]
        elif split == "val":
            image_paths, gt_paths = image_paths[75:85], gt_paths[75:85]
        elif split == "test":
            image_paths, gt_paths = image_paths[85:], gt_paths[85:]

    elif dataset == "sega":
        if split == "train":
            dchoice = "Rider"
        elif split == "val":
            dchoice = "Dongyang"
        elif split == "test":
            dchoice = "KiTS"
        else:
            raise ValueError(split)

        image_paths, gt_paths = medical.sega.get_sega_paths(path=path, data_choice=dchoice, download=False)

    elif dataset == "duke_liver":
        image_paths, gt_paths = medical.duke_liver.get_duke_liver_paths(path=path, split=split, download=False)

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
