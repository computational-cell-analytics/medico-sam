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
    """Task: Aorta Segmentation in CT Scans.

    We have three chunks of data: kits, rider, dongyang.
    - for validation:
    - for testing:
    """
    image_paths, gt_paths = medical.sega._get_sega_paths(
        path=os.path.join(ROOT, "sega"), data_choice=split_choice, download=False,
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
                fname=f"sega_{image_id}",
                save_dir=save_dir
            )

    # TODO: make val-test splits


def for_uwaterloo_skin(save_dir):
    """Task: Skin Lesion Segmentation in Dermoscopy Images

    - for validation:
    - for testing:
    """
    image_paths, gt_paths = medical.uwaterloo_skin._get_uwaterloo_skin_paths(
        path=os.path.join(ROOT, "uwaterloo_skin"), download=False,
    )

    # TODO: make val-test splits


def for_idrid(save_dir):
    """Task: Optic Disc Segmentation in Fundus Images

    - for validation:
    - for testing:
    """
    train_image_paths, train_gt_paths = medical.idrid._get_idrid_paths(
        path=os.path.join(ROOT, "idrid"), split="train", task="optic_disc", download=True,
    )

    test_image_paths, test_gt_paths = medical.idrid._get_idrid_paths(
        path=os.path.join(ROOT, "idrid"), split="train", task="optic_disc", download=True,
    )


def for_camus(save_dir, chamber_choice=2):
    """Task: Cardiac Structure Segmentation in US Scans.

    - for validation:
    - for testing:
    """
    image_paths, gt_paths = medical.camus._get_camus_paths(
        path=os.path.join(ROOT, "camus"), chamber=chamber_choice, download=True,
    )

    # TODO: check for 2 chamber and 4 chamber segmentations both

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
                fname=f"camus_{chamber_choice}_{image_id}",
                save_dir=save_dir
            )

    # TODO: make val and test splits


def for_montgomery(save_dir):
    """Task: Lung Segmentation in CXR Images.

    - for validation:
    - for testing:
    """
    image_paths, gt_paths = medical.montgomery._get_montgomery_paths(
        path=os.path.join(ROOT, "montgomery"), download=True,
    )

    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = read_image(image_path)
        gt = read_image(gt_path)

        image = resize_inputs(image)
        gt = resize_inputs(gt, is_label=True)

        show_images(image, gt)

        breakpoint()


def _preprocess_datasets(save_dir):
    # for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "kits"), split_choice="KiTS")
    # for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "rider"), split_choice="Rider")
    # for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "dongyang"), split_choice="Dongyang")

    # for_uwaterloo_skin(save_dir=os.path.join(save_dir, "uwaterloo_skin", "slices"))

    for_montgomery(save_dir=os.path.join(save_dir, "montgomery", "slices"))


def main():
    save_dir = "/scratch/share/cidas/cca/data"
    _preprocess_datasets(save_dir=save_dir)


if __name__ == "__main__":
    main()
