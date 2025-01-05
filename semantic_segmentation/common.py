import os

from torch_em.data.datasets import medical
from torch_em.data import MinInstanceSampler
from torch_em.transform.augmentation import get_augmentations


DATASETS_2D = ["oimhs", "isic", "dca1", "cbis_ddsm", "piccolo", "hil_toothseg"]
DATASETS_3D = ["osic_pulmofib", "duke_liver", "oasis", "lgg_mri", "leg_3d_us", "micro_usp"]

MODELS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/models"


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

    data_path = os.path.join(data_path, dataset_name)

    # 2D DATASETS
    if dataset_name == "oimhs":
        kwargs["sampler"] = MinInstanceSampler(min_num_instances=5)
        kwargs["transform"] = get_augmentations(ndim=2, transforms=["RandomHorizontalFlip"])
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

    elif dataset_name == "cbis_ddsm":
        kwargs["label_transform"] = LabelTrafoToBinary()
        train_loader = medical.get_cbis_ddsm_loader(path=data_path, batch_size=8, split="Train", task="Mass", **kwargs)
        val_loader = medical.get_cbis_ddsm_loader(path=data_path, batch_size=1, split="Val", task="Mass", **kwargs)

    elif dataset_name == "piccolo":
        kwargs["label_transform"] = LabelTrafoToBinary()
        train_loader = medical.get_piccolo_loader(path=data_path, batch_size=8, split="train", **kwargs)
        val_loader = medical.get_piccolo_loader(path=data_path, batch_size=1, split="validation", **kwargs)

    elif dataset_name == "hil_toothseg":
        kwargs["label_transform"] = LabelTrafoToBinary()
        train_loader = medical.get_hil_toothseg_loader(path=data_path, batch_size=8, split="train", **kwargs)
        val_loader = medical.get_hil_toothseg_loader(path=data_path, batch_size=1, split="val", **kwargs)

    # 3D DATASETS
    elif dataset_name == "osic_pulmofib":
        kwargs["transform"] = get_augmentations(ndim=3, transforms=["RandomHorizontalFlip3D", "RandomDepthicalFlip3D"])
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape, switch_last_axes=True)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(patch_shape, switch_last_axes=True, binary=False)
        train_loader = medical.get_osic_pulmofib_loader(
            path=data_path, batch_size=2, n_samples=100, split="train", **kwargs
        )
        val_loader = medical.get_osic_pulmofib_loader(path=data_path, batch_size=1, split="val", **kwargs)

    elif dataset_name == "duke_liver":
        kwargs["transform"] = get_augmentations(ndim=3, transforms=["RandomHorizontalFlip3D", "RandomDepthicalFlip3D"])
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape, switch_last_axes=True)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(desired_shape=patch_shape, switch_last_axes=True)
        train_loader = medical.get_duke_liver_loader(path=data_path, batch_size=2, split="train", **kwargs)
        val_loader = medical.get_duke_liver_loader(path=data_path, batch_size=1, split="val", **kwargs)

    elif dataset_name == "oasis":
        kwargs["sampler"] = MinInstanceSampler(min_num_instances=5)
        kwargs["raw_transform"] = RawTrafoFor3dInputs()
        train_loader = medical.get_oasis_loader(path=data_path, batch_size=2, split="train", **kwargs)
        val_loader = medical.get_oasis_loader(path=data_path, batch_size=1, split="val", **kwargs)

    elif dataset_name == "lgg_mri":
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(desired_shape=patch_shape)
        train_loader = medical.get_lgg_mri_loader(
            path=data_path, batch_size=2, split="train", channels="flair", n_samples=100, **kwargs
        )
        val_loader = medical.get_lgg_mri_loader(path=data_path, batch_size=1, split="val", channels="flair", **kwargs)

    elif dataset_name == "leg_3d_us":
        kwargs["sampler"] = MinInstanceSampler(min_num_instances=4)
        kwargs["raw_transform"] = RawTrafoFor3dInputs()
        train_loader = medical.get_leg_3d_us_loader(
            path=data_path, batch_size=2, split="train", n_samples=100, **kwargs
        )
        val_loader = medical.get_leg_3d_us_loader(path=data_path, batch_size=1, split="val", n_samples=10, **kwargs)

    elif dataset_name == "micro_usp":
        kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape)
        kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(desired_shape=patch_shape)
        train_loader = medical.get_micro_usp_loader(
            path=data_path, batch_size=2, split="train", n_samples=100, **kwargs
        )
        val_loader = medical.get_micro_usp_loader(path=data_path, batch_size=1, split="val", n_samples=10, **kwargs)

    else:
        raise ValueError(f"'{dataset_name}' is not a valid dataset name.")

    return train_loader, val_loader


def get_num_classes(dataset_name):
    if dataset_name in ["oimhs", "oasis"]:
        num_classes = 5
    elif dataset_name in ["osic_pulmofib", "leg_3d_us"]:
        num_classes = 4
    elif dataset_name in [
        "piccolo", "cbis_ddsm", "dca1", "isic", "hil_toothseg",  # 2d datasets
        "duke_liver", "lgg_mri", "micro_usp",  # 3d datasets
    ]:
        num_classes = 2
    else:
        raise ValueError

    return num_classes
