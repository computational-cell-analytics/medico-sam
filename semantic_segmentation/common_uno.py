import os

import numpy as np

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import util, medical
from torch_em.transform.augmentation import get_augmentations

from tukra.io import read_image


def get_dataloaders(patch_shape, data_path, dataset_name):
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
        "resize_inputs": True,
        "patch_shape": patch_shape,
        "num_workers": 16,
        "shuffle": True,
        "pin_memory": True,
        "sampler": MinInstanceSampler(),
        "raw_transform": sam_training.identity,
        "download": True,
    }

    train_raw_paths, train_label_paths = _get_data_paths(data_path, dataset_name, "train")
    val_raw_paths, val_label_paths = _get_data_paths(data_path, dataset_name, "val")

    # 2D DATASETS
    if dataset_name == "oimhs":
        kwargs["sampler"] = MinInstanceSampler(min_num_instances=5)
        kwargs["transform"] = get_augmentations(ndim=2, transforms=["RandomHorizontalFlip"])

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

    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)

    train_ds = torch_em.default_segmentation_dataset(
        raw_paths=train_raw_paths, raw_key=None, label_paths=train_label_paths, label_key=None, **ds_kwargs
    )
    val_ds = torch_em.default_segmentation_dataset(
        raw_paths=val_raw_paths, raw_key=None, label_paths=val_label_paths, label_key=None, **ds_kwargs
    )

    train_loader = torch_em.get_data_loader(train_ds, batch_size=2, **loader_kwargs)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=1, **loader_kwargs)

    return train_loader, val_loader


def _get_data_paths(data_path, dataset_name, split):
    data_path = os.path.join(data_path, dataset_name)

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

    input_paths = get_paths[dataset_name]()
    if isinstance(input_paths, tuple):
        raw_paths, label_paths = input_paths
    else:
        raw_paths = label_paths = input_paths

    for raw_path, label_path in zip(raw_paths, label_paths):
        raw = read_image(raw_path)
        label = read_image(label_path)
        label = np.round(label).astype(int)  # Ensuring label ids as integers.

        breakpoint()

    breakpoint()

    if split == "train":
        raw_paths = ...
        label_paths = ...
    elif split == "val":
        raw_paths = ...
        label_paths = ...
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    return raw_paths, label_paths
