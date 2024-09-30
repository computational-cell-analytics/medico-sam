import os
from glob import glob
from natsort import natsorted

import torch_em
from torch_em.data.datasets import util
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import medical

import micro_sam.training as sam_training

from medico_sam.transform.raw import RawTrafoFor3dInputs, RawResizeTrafoFor3dInputs
from medico_sam.transform.label import LabelTrafoToBinary, LabelResizeTrafoFor3dInputs


DATASETS = [
    # 2d datasets
    "oimhs", "isic", "dca1", "cbis_ddsm", "drive", "piccolo",
    # 3d datasets
    # "btcv", "osic_pulmofib", "sega", "duke_liver"
]

MODELS_ROOT = "/scratch/share/cidas/cca/models"


def get_dataloaders(patch_shape, data_path, dataset_name):
    """This returns the medical data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/medical/

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    kwargs = {
        "resize_inputs": True,
        "patch_shape": patch_shape,
        "num_workers": 16,
        "shuffle": True,
        "pin_memory": True,
        "sampler": MinInstanceSampler(),
    }

    data_path = os.path.join(data_path, dataset_name)

    if dataset_name == "oimhs":
        kwargs["sampler"] = MinInstanceSampler(min_num_instances=5)
        kwargs["raw_transform"] = sam_training.identity
        train_loader = medical.get_oimhs_loader(path=data_path, batch_size=8, split="train", **kwargs)
        val_loader = medical.get_oimhs_loader(path=data_path, batch_size=1, split="val", **kwargs)

        train_loader.dataset.max_sampling_attempts = 10000
        val_loader.dataset.max_sampling_attempts = 10000

    elif dataset_name == "dca1":
        kwargs["label_transform"] = LabelTrafoToBinary()
        kwargs["raw_transform"] = sam_training.identity
        train_loader = medical.get_dca1_loader(path=data_path, batch_size=8, split="train", n_samples=400, **kwargs)
        val_loader = medical.get_dca1_loader(path=data_path, batch_size=1, split="val", **kwargs)

    elif dataset_name == "drive":
        kwargs["label_transform"] = LabelTrafoToBinary()
        kwargs["raw_transform"] = sam_training.identity
        train_loader = medical.get_drive_loader(path=data_path, batch_size=8, split="train", n_samples=400, **kwargs)
        val_loader = medical.get_drive_loader(path=data_path, batch_size=1, split="val", n_samples=15, **kwargs)

    elif dataset_name == "isic":
        kwargs["label_transform"] = LabelTrafoToBinary()
        kwargs["raw_transform"] = sam_training.identity
        train_loader = medical.get_isic_loader(path=data_path, batch_size=8, split="train", **kwargs)
        val_loader = medical.get_isic_loader(path=data_path, batch_size=1, split="val", **kwargs)

    elif dataset_name == "piccolo":
        kwargs["label_transform"] = LabelTrafoToBinary()
        kwargs["raw_transform"] = sam_training.identity
        train_loader = medical.get_piccolo_loader(path=data_path, batch_size=8, split="train", **kwargs)
        val_loader = medical.get_piccolo_loader(path=data_path, batch_size=1, split="validation", **kwargs)

    elif dataset_name == "duke_liver":
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(desired_shape=patch_shape)
        train_loader = medical.get_duke_liver_loader(path=data_path, batch_size=2, split="train", **kwargs)
        val_loader = medical.get_duke_liver_loader(path=data_path, batch_size=1, split="val", **kwargs)

    elif dataset_name == "sega":
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(desired_shape=patch_shape)
        train_loader = medical.get_sega_loader(path=data_path, batch_size=2, data_choice="Rider", **kwargs)
        val_loader = medical.get_sega_loader(path=data_path, batch_size=1, data_choice="Dongyang", **kwargs)

    else:
        if dataset_name == "btcv":
            data_path = "/scratch/share/cidas/cca/nnUNetv2/nnUNet_raw/Dataset301_BTCV/"
            print("The path to 'BTCV' dataset has been hard-coded at the moment.")

            kwargs["raw_transform"] = RawTrafoFor3dInputs()
            kwargs["sampler"] = MinInstanceSampler(min_num_instances=8)
            ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
            ds_kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
                ds_kwargs, patch_shape, resize_inputs=True, resize_kwargs={"patch_shape": patch_shape, "is_rgb": False}
            )
            ds_kwargs = {"raw_key": "data", "label_key": "data", "ndim": 3, "is_seg_dataset": True, **kwargs}

        elif dataset_name == "osic_pulmofib":
            data_path = "/scratch/share/cidas/cca/nnUNetv2/nnUNet_raw/Dataset303_OSICPulmoFib/"
            print("The path to 'OSIC PulmoFib' dataset has been hard-coded at the moment.")

            kwargs["raw_transform"] = RawTrafoFor3dInputs()
            ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
            ds_kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
                ds_kwargs, patch_shape, resize_inputs=True, resize_kwargs={"patch_shape": patch_shape, "is_rgb": False}
            )
            ds_kwargs = {"raw_key": "data", "label_key": "data", "ndim": 3, "is_seg_dataset": True, **ds_kwargs}

        elif dataset_name == "cbis_ddsm":
            data_path = "/scratch/share/cidas/cca/nnUNetv2/nnUNet_raw/Dataset206_CBISDDSM/"
            print("The path to 'CBIS-DDSM' dataset has been hard-coded at the moment.")

            kwargs.pop("resize_inputs")
            kwargs["label_transform"] = LabelTrafoToBinary()
            ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
            ds_kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
                ds_kwargs, patch_shape, resize_inputs=True, resize_kwargs={"patch_shape": patch_shape, "is_rgb": False}
            )
            ds_kwargs = {"raw_key": None, "label_key": None, "is_seg_dataset": False, **ds_kwargs}
            ds_kwargs["patch_shape"] = patch_shape

        else:
            raise ValueError

        _extension = ".tif" if dataset_name == "cbis_ddsm" else ".nii.gz"
        train_image_paths = natsorted(glob(os.path.join(data_path, "imagesTr", f"*_train_0000{_extension}")))
        train_gt_paths = natsorted(glob(os.path.join(data_path, "labelsTr", f"*_train{_extension}")))
        val_image_paths = natsorted(glob(os.path.join(data_path, "imagesTr", f"*_val_0000{_extension}")))
        val_gt_paths = natsorted(glob(os.path.join(data_path, "labelsTr", f"*_val{_extension}")))

        train_dataset = torch_em.default_segmentation_dataset(
            raw_paths=train_image_paths, label_paths=train_gt_paths, **ds_kwargs
        )
        val_dataset = torch_em.default_segmentation_dataset(
            raw_paths=val_image_paths, label_paths=val_gt_paths, **ds_kwargs
        )

        train_loader = torch_em.get_data_loader(dataset=train_dataset, batch_size=8, **loader_kwargs)
        val_loader = torch_em.get_data_loader(dataset=val_dataset, batch_size=1, **loader_kwargs)

    return train_loader, val_loader


def get_num_classes(dataset_name):
    if dataset_name == "btcv":
        num_classes = 14
    elif dataset_name == "oimhs":
        num_classes = 5
    elif dataset_name == "osic_pulmofib":
        num_classes = 4
    elif dataset_name in [
        "piccolo", "cbis_ddsm", "dca1", "drive", "isic",  # 2d datasets
        "duke_liver", "sega",  # 3d datasets
    ]:
        num_classes = 2
    else:
        raise ValueError

    return num_classes
