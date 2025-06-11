import os
from pathlib import Path
from typing import Union, Optional, Literal

import numpy as np

from elf.io import open_file

from torch_em.data.datasets.medical.sa_med2d import get_sa_med2d_paths, SHARD_SIZE


BRAIN_DATASETS = [
    "BrainTumour",  # brain tumor in MRI
    "BraTS2013",  # brain tumour in MRI
    "BraTS2015",  # brain tumour in MRI
    "BraTS2018",  # brain tumour in MRI
    "BraTS2019",  # brain tumour in MRI
    "BraTS2020",  # brain tumour in MRI
    "BraTS2021",  # brain tumour in MRI
    "Brain_PTM",  # white matter tracts in brain MRI
    "cranium",  # cranial segmentation in CT
    "Instance22",  # intracranial hemorrhage in NC-CT
    "ISLES_SPES",  # ischemic stroke lesion in brain MRI
    "ISLES_SISS",  # ischemic stroke lesion in brain MRI
    "ISLES2016",  # ischemic stroke lesion in brain MRI
    "ISLES2017",  # ischemic stroke lesion in brain MRI
    "ISLES2018",  # ischemic stroke in brain CT
    "ISLES2022",  # ischemic stroke in brain MRI
    "LongitudinalMultipleSclerosisLesionSegmentation",  # MS lesion in FLAIR-MRI
]


def get_sa_med2d_rois(
    path: Union[str, os.PathLike],
    split: Literal["train", "val"],
    fraction: Optional[float] = None,
    min_samples: int = 20,
):
    """Create RoIs for the SA-Med2d dataset.

    Args:
        path: The filepath where the dataset is stored.
        split: The choice of data split. Either 'train' or 'val'.
        fraction: The fraction of dataset to choose. By default, returns the entire dataset.
        min_samples: The minimum number of samples to reserve, if dataset is too small.

    Returns:
        The RoIs per sub-dataset for the entire dataset.
    """
    assert split in ["train", "val"], split

    # Get the h5 files in the same order as provided to the dataset.
    input_paths = get_sa_med2d_paths(path=path)
    sample_sizes = {
        Path(p).stem: open_file(p, mode="r")["labels"].shape[0] for p in input_paths
    }

    # We undersample the brain-related datasets: down to only 10% per bulky datasets!
    sample_sizes = [
        (size * 0.1) if size == SHARD_SIZE and name[:-3] in BRAIN_DATASETS
        else size for name, size in sample_sizes.items()
    ]

    # We take the first 80% samples and set them for the 'train' split.
    train_sample_sizes = [int(s * 0.8) for s in sample_sizes]

    # We take the remaining 20% samples and set them for the 'val' split.
    val_sample_sizes = [int(s - ts) for s, ts in zip(sample_sizes, train_sample_sizes)]

    # We get only a fraction of the dataset, if desired.
    if fraction is not None:
        assert fraction > 0 and fraction <= 1, fraction
        train_sample_sizes = [int(s * fraction) if s > min_samples else s for s in train_sample_sizes]
        val_sample_sizes = [int(s * fraction) if s > min_samples else s for s in val_sample_sizes]

    # Now, we can finally get the rois.
    train_rois = [(np.s_[:ts],) for ts in train_sample_sizes]
    val_rois = [(np.s_[ts: ts+vs],) for ts, vs in zip(train_sample_sizes, val_sample_sizes)]

    if split == "train":
        return train_rois
    else:
        return val_rois
