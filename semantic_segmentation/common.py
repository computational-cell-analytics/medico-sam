import os

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import medical
from torch_em.transform.augmentation import get_augmentations

import micro_sam.training as sam_training

from medico_sam.transform.raw import RawTrafoFor3dInputs, RawResizeTrafoFor3dInputs
from medico_sam.transform.label import LabelTrafoToBinary, LabelResizeTrafoFor3dInputs


DATASETS_2D = [
    "oimhs", "isic", "dca1", "cbis_ddsm", "drive", "piccolo", "siim_acr", "hil_toothseg", "covid_qu_ex"
]
DATASETS_3D = [
    "btcv", "osic_pulmofib", "sega", "duke_liver"
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
        "raw_transform": sam_training.identity,
    }

    data_path = os.path.join(data_path, dataset_name)

    # 2D DATASETS
    if dataset_name == "oimhs":
        kwargs["sampler"] = MinInstanceSampler(min_num_instances=5)
        train_loader = medical.get_oimhs_loader(path=data_path, batch_size=8, split="train", **kwargs)
        val_loader = medical.get_oimhs_loader(path=data_path, batch_size=1, split="val", **kwargs)

        train_loader.dataset.max_sampling_attempts = 10000
        val_loader.dataset.max_sampling_attempts = 10000

    elif dataset_name == "isic":
        kwargs["label_transform"] = LabelTrafoToBinary()
        train_loader = medical.get_isic_loader(path=data_path, batch_size=8, split="train", **kwargs)
        val_loader = medical.get_isic_loader(path=data_path, batch_size=1, split="val", **kwargs)

    elif dataset_name == "dca1":
        kwargs["label_transform"] = LabelTrafoToBinary()
        train_loader = medical.get_dca1_loader(path=data_path, batch_size=8, split="train", n_samples=400, **kwargs)
        val_loader = medical.get_dca1_loader(path=data_path, batch_size=1, split="val", **kwargs)

    elif data_path == "cbis_ddsm":
        kwargs["label_transform"] = LabelTrafoToBinary()
        train_loader = medical.get_cbis_ddsm_loader(path=data_path, batch_size=8, split="Train", task="Mass", **kwargs)
        val_loader = medical.get_cbis_ddsm_loader(path=data_path, batch_size=8, split="Val", task="Mass", **kwargs)

    elif dataset_name == "drive":
        kwargs["label_transform"] = LabelTrafoToBinary()
        train_loader = medical.get_drive_loader(path=data_path, batch_size=8, split="train", n_samples=400, **kwargs)
        val_loader = medical.get_drive_loader(path=data_path, batch_size=1, split="val", n_samples=15, **kwargs)

    elif dataset_name == "piccolo":
        kwargs["label_transform"] = LabelTrafoToBinary()
        train_loader = medical.get_piccolo_loader(path=data_path, batch_size=8, split="train", **kwargs)
        val_loader = medical.get_piccolo_loader(path=data_path, batch_size=1, split="validation", **kwargs)

    elif dataset_name == "siim_acr":
        kwargs["label_transform"] = LabelTrafoToBinary()
        train_loader = medical.get_siim_acr_loader(path=data_path, batch_size=8, split="train", **kwargs)
        val_loader = medical.get_siim_acr_loader(path=data_path, batch_size=1, split="val", **kwargs)

    elif dataset_name == "hil_toothseg":
        kwargs["label_transform"] = LabelTrafoToBinary()
        train_loader = medical.get_hil_toothseg_loader(path=data_path, batch_size=8, split="train", **kwargs)
        val_loader = medical.get_hil_toothseg_loader(path=data_path, batch_size=1, split="val", **kwargs)

    elif dataset_name == "covid_qu_ex":
        kwargs["label_transform"] = LabelTrafoToBinary()
        train_loader = medical.get_covid_qu_ex_loader(
            path=data_path, batch_size=8, split="train", task="lung", **kwargs
        )
        val_loader = medical.get_covid_qu_ex_loader(path=data_path, batch_size=1, split="val", task="lung", **kwargs)

    # 3D DATASETS
    elif dataset_name == "curvas":
        ...

    elif dataset_name == "osic_pulmofib":
        ...

    elif dataset_name == "sega":
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(desired_shape=patch_shape)
        train_loader = medical.get_sega_loader(path=data_path, batch_size=2, data_choice="Rider", **kwargs)
        val_loader = medical.get_sega_loader(path=data_path, batch_size=1, data_choice="Dongyang", **kwargs)

    elif dataset_name == "duke_liver":
        kwargs["transform"] = get_augmentations(ndim=3, transforms=["RandomHorizontalFlip3D", "RandomDepthicalFlip3D"])
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape, switch_last_axes=True)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(desired_shape=patch_shape, switch_last_axes=True)
        train_loader = medical.get_duke_liver_loader(path=data_path, batch_size=2, split="train", **kwargs)
        val_loader = medical.get_duke_liver_loader(path=data_path, batch_size=1, split="val", **kwargs)

    elif dataset_name == "toothfairy2":
        ...

    elif dataset_name == "segthy":
        ...

    elif dataset_name == "oasis":
        ...

    elif dataset_name == "lgg_mri":
        ...

    elif dataset_name == "leg_3d_us":
        ...

    elif dataset_name == "micro_usp":
        ...

    else:
        raise ValueError

    return train_loader, val_loader


def get_num_classes(dataset_name):
    if dataset_name == "btcv":
        num_classes = 14
    elif dataset_name == "oimhs":
        num_classes = 5
    elif dataset_name == "osic_pulmofib":
        num_classes = 4
    elif dataset_name in [
        "piccolo", "cbis_ddsm", "dca1", "drive", "isic", "siim_acr", "hil_toothseg", "covid_qu_ex",  # 2d datasets
        "duke_liver", "sega",  # 3d datasets
    ]:
        num_classes = 2
    else:
        raise ValueError

    return num_classes
