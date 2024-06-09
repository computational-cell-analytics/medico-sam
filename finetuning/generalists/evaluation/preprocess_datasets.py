import os
from tqdm import tqdm
from pathlib import Path

import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
from skimage.transform import resize

from tukra.utils import read_image

from torch_em.data.datasets import medical


ROOT = "/scratch/share/cidas/cca/data"


def has_foreground(label):
    if len(np.unique(label)) > 1:
        return True
    else:
        return False


def resize_inputs(image, patch_shape=(1024, 1024), is_label=False):
    if is_label:
        kwargs = {"order": 0,  "anti_aliasing": False}
    else:  # we use the default settings for float data
        kwargs = {}

    image = resize(
        image=image,
        output_shape=patch_shape,
        preserve_range=True,
        **kwargs
    ).astype(image.dtype)

    if not is_label:
        image = np.stack([image] * 3, axis=-1)

    return image


def get_valid_slices_per_volume(image, gt, fname, save_dir, visualize=False, overwrite_images=True):
    """This function assumes the volumes to be channels first: Z * Y * X
    """
    assert image.shape == gt.shape

    image_dir = os.path.join(save_dir, "images")
    gt_dir = os.path.join(save_dir, "ground_truth")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for i, (image_slice, gt_slice) in enumerate(zip(image, gt)):
        image_path = os.path.join(image_dir, f"{fname}_{i:05}.tif")
        gt_path = os.path.join(gt_dir, f"{fname}_{i:05}.tif")

        if os.path.exists(image_path) and os.path.exists(gt_path) and overwrite_images:
            continue

        if has_foreground(gt_slice):
            image_slice = resize_inputs(image_slice)
            gt_slice = resize_inputs(gt_slice, is_label=True)

            if visualize:
                show_images(image_slice, gt_slice)

            imageio.imwrite(image_path, image_slice, compression="zlib")
            imageio.imwrite(gt_path, gt_slice, compression="zlib")


def show_images(*images, save_path=None):
    fig, ax = plt.subplots(1, len(images), figsize=(10, 10))

    ax[0].imshow(images[0], cmap="gray")
    for idx, img in enumerate(images[1:], start=1):
        ax[idx].imshow(img, cmap="gray")

    plt.savefig("./save_fig.png" if save_path is None else save_path)
    plt.close()


#
# Medical Imaging Datasets
#

# TODO: it would make more sense to make per-dataset task wise partitions - makes sense to report results like that.


def for_sega(save_dir, split_choice):
    """We have three chunks of data: kits, rider, dongyang.
    - for validation:
    - for testing:
    """
    # first, we convert all volumes to valid slices
    image_paths, gt_paths = medical.sega._get_sega_paths(
        path=os.path.join(ROOT, "sega"), data_choice=split_choice, download=True
    )

    for image_path, gt_path in zip(image_paths, gt_paths):
        image = read_image(image_path)
        gt = read_image(gt_path)

        image_id = Path(image_path).stem

        # make channels first
        image, gt = image.transpose(2, 0, 1), gt.transpose(2, 0, 1)

        for islice, gslice in tqdm(zip(image, gt), total=image.shape[0], desc=f"Processing '{image_id}'"):
            get_valid_slices_per_volume(
                image=islice,
                gt=gslice,
                fname=image_id,
                save_dir=save_dir
            )


def _preprocess_datasets(save_dir):
    for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "kits"), split_choice="KiTS")
    for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "rider"), split_choice="Rider")
    for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "dongyang"), split_choice="Dongyang")


def main():
    save_dir = "/scratch/share/cidas/cca/data"
    _preprocess_datasets(save_dir=save_dir)


if __name__ == "__main__":
    main()
