import os
from tqdm import tqdm

import json
import numpy as np

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import util, medical
from torch_em.transform.augmentation import get_augmentations

from tukra.io import read_image


def get_dataloaders(patch_shape, data_path, dataset_name, view):
    """This returns the medical data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/medical/

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive.
    """
    import micro_sam.training as sam_training

    from medico_sam.transform.raw import RawTrafoFor3dInputs, RawResizeTrafoFor3dInputs
    from medico_sam.transform.label import LabelTrafoToBinary, LabelResizeTrafoFor3dInputs

    kwargs = {
        "sampler": MinInstanceSampler(),
        "raw_transform": sam_training.identity,
        "n_samples": 100,
    }

    train_raw_paths, train_label_paths = _get_data_paths(data_path, dataset_name, "train", view)
    val_raw_paths, val_label_paths = _get_data_paths(data_path, dataset_name, "val", view)

    # 2D DATASETS
    if dataset_name == "oimhs":
        kwargs["sampler"] = MinInstanceSampler(min_num_instances=5)
        kwargs["transform"] = get_augmentations(ndim=2, transforms=["RandomHorizontalFlip"])
        kwargs["is_seg_dataset"] = False

    elif dataset_name == "isic":
        kwargs["label_transform"] = LabelTrafoToBinary()

    elif dataset_name == "dca1":
        kwargs["label_transform"] = LabelTrafoToBinary()

    elif dataset_name == "cbis_ddsm":
        kwargs["label_transform"] = LabelTrafoToBinary()

    elif dataset_name == "piccolo":
        kwargs["label_transform"] = LabelTrafoToBinary()

    elif dataset_name == "hil_toothseg":
        kwargs["label_transform"] = LabelTrafoToBinary()

    # 3D DATASETS
    elif dataset_name == "osic_pulmofib":
        kwargs["transform"] = get_augmentations(ndim=3, transforms=["RandomHorizontalFlip3D", "RandomDepthicalFlip3D"])
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape, switch_last_axes=True)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(patch_shape, switch_last_axes=True, binary=False)

    elif dataset_name == "duke_liver":
        kwargs["transform"] = get_augmentations(ndim=3, transforms=["RandomHorizontalFlip3D", "RandomDepthicalFlip3D"])
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape, switch_last_axes=True)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(desired_shape=patch_shape, switch_last_axes=True)

    elif dataset_name == "oasis":
        kwargs["sampler"] = MinInstanceSampler(min_num_instances=5)
        kwargs["raw_transform"] = RawTrafoFor3dInputs()

    elif dataset_name == "lgg_mri":
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(desired_shape=patch_shape)

    elif dataset_name == "leg_3d_us":
        kwargs["sampler"] = MinInstanceSampler(min_num_instances=4)
        kwargs["raw_transform"] = RawTrafoFor3dInputs()

    elif dataset_name == "micro_usp":
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(desired_shape=patch_shape)

    else:
        raise ValueError(f"'{dataset_name}' is not a valid dataset name.")

    # Ensure resizing inputs.
    kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
        kwargs=kwargs,
        patch_shape=patch_shape,
        resize_inputs=True,
        resize_kwargs={"patch_shape": patch_shape, "is_rgb": False},
    )

    # Get the datasets
    def _get_dataset(rpaths, lpaths):
        ds = torch_em.default_segmentation_dataset(
            raw_paths=[rpaths], raw_key=None, label_paths=[lpaths], label_key=None, patch_shape=patch_shape, **kwargs
        )
        return ds

    train_ds = _get_dataset(train_raw_paths, train_label_paths)
    val_ds = _get_dataset(val_raw_paths, val_label_paths)

    # Get the dataloaders
    train_loader = torch_em.get_data_loader(dataset=train_ds, batch_size=2, num_workers=16)
    val_loader = torch_em.get_data_loader(dataset=val_ds, batch_size=1, num_workers=16)

    return train_loader, val_loader


def _get_data_paths(data_path, dataset_name, split, view):
    data_path = os.path.join(data_path, dataset_name)

    # Let's check if the files are decided and exist already.
    target_dir = os.path.join(data_path, "uno")
    json_file = os.path.join(target_dir, f"{dataset_name}_{split}.json")
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            fpaths = json.load(f)

    else:
        os.makedirs(target_dir, exist_ok=True)

        get_paths = {
            # 2d
            "oimhs": lambda: medical.oimhs.get_oimhs_paths(path=data_path, split=split, download=True),
            # "isic": lambda: medical.isic.get_isic_paths(path=data_path, split=split, download=True),
            # "dca1": lambda: medical.dca1.get_dca1_paths(path=data_path, split=split, download=True),
            # "cbis_ddsm": lambda: medical.cbis_ddsm.get_cbis_ddsm_paths(
            #     path=data_path, split=split.title(), task="Mass", download=True,
            # ),
            # "piccolo": medical.piccolo.get_piccolo_paths(path=data_path, split="validation" if split == "val" else split),
            # "hil_toothseg": medical.hil_toothseg.get_hil_toothseg_paths(path=data_path, split=split, download=True),
            # # 3d
            # "osic_pulmofib": lambda: medical.osic_pulmofib.get_osic_pulmofib_paths(
            #     path=data_path, split=split, download=True
            # ),
            # "duke_liver": lambda: medical.duke_liver.get_duke_liver_paths(path=data_path, split=split, download=True),
            # "oasis": lambda: medical.oasis.get_oasis_paths(path=data_path, split=split, download=True),
            # "lgg_mri": lambda: medical.lgg_mri.get_lgg_mri_paths(path=data_path, split=split, download=True),
            # "leg_3d_us": lambda: medical.leg_3d_us.get_leg_3d_us_paths(path=data_path, split=split, download=True),
            # "micro_usp": lambda: medical.micro_usp.get_micro_usp_paths(path=data_path, split=split, download=True),
        }

        get_ids = {
            # 2d
            "oimhs": 5, "isic": 2, "dca1": 2, "cbis_ddsm": 2, "piccolo": 2, "hil_toothseg": 2,
            # 3d
            "osic_pulmofib": 4, "duke_liver": 2, "oasis": 5, "lgg_mri": 2, "leg_3d_us": 4, "micro_usp": 2,
        }

        input_paths = get_paths[dataset_name]()
        if isinstance(input_paths, tuple):
            raw_paths, label_paths = input_paths
        else:
            raw_paths = label_paths = input_paths

        fpaths = {}
        for raw_path, label_path in tqdm(
            zip(raw_paths, label_paths), total=len(raw_paths), desc="Extracting images",
        ):
            raw = read_image(raw_path)
            label = read_image(label_path)
            label = np.round(label).astype(int)  # Ensuring label ids as integers.

            sampler = MinInstanceSampler(min_num_instances=get_ids[dataset_name])
            if sampler(raw, label):
                if view:
                    import napari
                    v = napari.Viewer()
                    v.add_image(raw, name="Image")
                    v.add_labels(label, name="Label")
                    napari.run()

                fpaths[split] = [os.path.relpath(raw_path, data_path), os.path.relpath(label_path, data_path)]
                break

        with open(json_file, "w") as f:
            json.dump(fpaths, f, indent=4)

    raw_path, label_path = fpaths[split]
    raw_path, label_path = os.path.join(data_path, raw_path), os.path.join(data_path, label_path)

    return raw_path, label_path
