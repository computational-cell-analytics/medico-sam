import os
import shutil
import random
from tqdm import tqdm
from glob import glob
from pathlib import Path

import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
from skimage.transform import resize

from tukra.utils import read_image

from torch_em.data.datasets import medical
from torch_em.transform.raw import normalize


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
        if image.ndim == 2:
            image = normalize(image)
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            assert image.shape[-1] == 3
            image = normalize(image, axis=(0, 1))

        image = image * 255

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


def _get_val_test_splits(save_dir, val_fraction, fname_ext=None):
    assert fname_ext is not None
    image_paths = sorted(glob(os.path.join(save_dir, "images", f"{fname_ext}*.tif")))
    gt_paths = sorted(glob(os.path.join(save_dir, "ground_truth", f"{fname_ext}*.tif")))

    assert len(image_paths) == len(gt_paths)

    if val_fraction < 1:  # means a percentage of images
        assert val_fraction > 0
        n_val = len(image_paths) * val_fraction
    else:  # means n number of images
        n_val = val_fraction

    val_image_paths = random.sample(image_paths, k=n_val)
    val_gt_paths = [
        os.path.join(save_dir, "ground_truth", os.path.split(vpath)[-1]) for vpath in val_image_paths
    ]

    test_image_paths = [tpath for tpath in image_paths if tpath not in val_image_paths]
    test_gt_paths = [
        os.path.join(save_dir, "ground_truth", os.path.split(tpath)[-1]) for tpath in test_image_paths
    ]

    def _move_images(split, image_paths, gt_paths):
        trg_image_dir = os.path.join(save_dir, split, "images")
        trg_gt_dir = os.path.join(save_dir, split, "ground_truth")

        os.makedirs(trg_image_dir, exist_ok=True)
        os.makedirs(trg_gt_dir, exist_ok=True)

        for image_path, gt_path in zip(image_paths, gt_paths):
            trg_image_path = os.path.join(trg_image_dir, os.path.split(image_path)[-1])
            trg_gt_path = os.path.join(trg_gt_dir, os.path.split(gt_path)[-1])

            shutil.move(src=image_path, dst=trg_image_path)
            shutil.move(src=gt_path, dst=trg_gt_path)

    _move_images(split="val", image_paths=val_image_paths, gt_paths=val_gt_paths)
    _move_images(split="test", image_paths=test_image_paths, gt_paths=test_gt_paths)

    shutil.rmtree(path=os.path.join(save_dir, "images"))
    shutil.rmtree(path=os.path.join(save_dir, "ground_truth"))


def _check_preprocessing(save_dir):
    return os.path.exists(os.path.join(save_dir, "val")) and os.path.exists(os.path.join(save_dir, "test"))


def convert_simple_datasets(image_paths, gt_paths, save_dir, fname_ext):
    image_dir = os.path.join(save_dir, "images")
    gt_dir = os.path.join(save_dir, "ground_truth")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for idx, (image_path, gt_path) in tqdm(enumerate(zip(image_paths, gt_paths)), total=len(image_paths)):
        image_id = Path(image_path).stem

        trg_image_path = os.path.join(image_dir, f"{fname_ext}{image_id}_{idx:05}.tif")
        trg_gt_path = os.path.join(gt_dir, f"{fname_ext}{image_id}_{idx:05}.tif")

        if os.path.exists(trg_image_path) and os.path.exists(trg_gt_path):
            continue

        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)

        if has_foreground(gt):
            image = resize_inputs(image)
            gt = resize_inputs(gt, is_label=True)

            if gt.dtype == "bool":  # for uwaterloo
                gt = gt.astype("uint8")

            imageio.imwrite(trg_image_path, image, compression="zlib")
            imageio.imwrite(trg_gt_path, gt, compression="zlib")


#
# Medical Imaging Datasets
#

# TODO: it would make more sense to make per-dataset task wise partitions - makes sense to report results like that.


def for_sega(save_dir, split_choice):
    """Task: Aorta Segmentation in CT Scans.

    We have three chunks of data: kits, rider, dongyang.
    - for validation: 50*3 (respectively)
    - for testing: 4540, 4097, 2988 (respectively)
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

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

    _get_val_test_splits(save_dir=save_dir, fname_ext="sega_", val_fraction=50)


def for_uwaterloo_skin(save_dir):
    """Task: Skin Lesion Segmentation in Dermoscopy Images

    We have two sets of data for the same task.
    - for validation: 10
    - for testing: 196
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.uwaterloo_skin._get_uwaterloo_skin_paths(
        path=os.path.join(ROOT, "uwaterloo_skin"), download=False,
    )

    fext = "uwaterloo_skin_"
    convert_simple_datasets(image_paths=image_paths, gt_paths=gt_paths, save_dir=save_dir, fname_ext=fext)
    _get_val_test_splits(save_dir=save_dir, val_fraction=10, fname_ext=fext)


def for_idrid(save_dir):
    """Task: Optic Disc Segmentation in Fundus Images

    - for validation: 5
    - for testing: 76
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    train_image_paths, train_gt_paths = medical.idrid._get_idrid_paths(
        path=os.path.join(ROOT, "idrid"), split="train", task="optic_disc", download=False,
    )

    test_image_paths, test_gt_paths = medical.idrid._get_idrid_paths(
        path=os.path.join(ROOT, "idrid"), split="test", task="optic_disc", download=False,
    )

    train_image_paths.extend(test_image_paths)
    train_gt_paths.extend(test_gt_paths)

    fext = "idrid_"
    convert_simple_datasets(image_paths=train_image_paths, gt_paths=train_gt_paths, save_dir=save_dir, fname_ext=fext)
    _get_val_test_splits(save_dir=save_dir, val_fraction=5, fname_ext=fext)


def for_camus(save_dir, chamber_choice):
    """Task: Cardiac Structure Segmentation in Echocardiography Scans.

    - for validation:
    - for testing:
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.camus._get_camus_paths(
        path=os.path.join(ROOT, "camus"), chamber=chamber_choice, download=True,
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
                fname=f"camus_{image_id}",
                save_dir=save_dir
            )

    _get_val_test_splits(save_dir=save_dir, val_fraction=50, fname_ext="camus_")


def for_montgomery(save_dir):
    """Task: Lung Segmentation in CXR Images.

    - for validation: 10
    - for testing: 128
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.montgomery._get_montgomery_paths(
        path=os.path.join(ROOT, "montgomery"), download=False,
    )

    fext = "montgomery_"
    convert_simple_datasets(image_paths=image_paths, gt_paths=gt_paths, save_dir=save_dir, fname_ext=fext)
    _get_val_test_splits(save_dir=save_dir, val_fraction=10, fname_ext=fext)


def _preprocess_datasets(save_dir):
    for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "kits"), split_choice="KiTS")
    for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "rider"), split_choice="Rider")
    for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "dongyang"), split_choice="Dongyang")
    for_uwaterloo_skin(save_dir=os.path.join(save_dir, "uwaterloo_skin", "slices"))
    for_idrid(save_dir=os.path.join(save_dir, "idrid", "slices"))

    # for_camus(save_dir=os.path.join(save_dir, "camus", "slices", "2ch"), chamber_choice=2)
    # for_camus(save_dir=os.path.join(save_dir, "camus", "slices", "4ch"), chamber_choice=4)

    for_montgomery(save_dir=os.path.join(save_dir, "montgomery", "slices"))


def main():
    save_dir = "/scratch/share/cidas/cca/data"
    _preprocess_datasets(save_dir=save_dir)


if __name__ == "__main__":
    main()
