import os
from glob import glob

import numpy as np
import imageio.v3 as imageio

from tukra.utils import read_image

from torch_em.data.datasets import medical


ROOT = "/scratch/share/cidas/cca/data"


def has_foreground(label):
    if len(np.unique(label)) > 1:
        return True
    else:
        return False


def _get_valid_slices_per_volume(image, gt, fname, save_dir):
    """This function assumes the volumes to be channels first: Z * Y * X
    """
    assert image.shape == gt.shape

    for image_slice, gt_slice in zip(image, gt):
        image_path = ...
        gt_path = ...

        if has_foreground(gt_slice):
            imageio.imwrite(image_path, image_slice, compression="zlib")
            imageio.imwrite(gt_path, gt_slice, compression="zlib")


#
# Medical Imaging Datasets
#

# TODO: it would make more sense to make per-dataset task wise partitions - makes sense to report results like that.


def for_sega(save_dir):
    """We have three chunks of data: kits, rider, dongyang.
    - for validation:
    - for testing:
    """
    # first, we convert all volumes to valid slices
    image_paths, gt_paths = medical.sega._get_sega_paths(path=ROOT, data_choice=None, download=False)

    for image_path, gt_path in zip(image_paths, gt_paths):
        image = read_image(image_path)
        gt = read_image(gt_path)

        # make channels first
        image, gt = image.transpose(2, 0, 1), gt.transpose(2, 0, 1)

        _get_valid_slices_per_volume(image=image, gt=gt, fname=image_path, save_dir=save_dir)
