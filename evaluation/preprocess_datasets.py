import os
import shutil
import random
from tqdm import tqdm
from glob import glob
from pathlib import Path

import numpy as np
import nibabel as nib
import imageio.v3 as imageio
import matplotlib.pyplot as plt
from skimage.transform import resize

# TODO: will make a release of tukra soon
from tukra.utils import read_image

from torch_em.data.datasets import medical
from torch_em.transform.raw import normalize


ROOT = "/scratch/share/cidas/cca/data"


def has_foreground(label, min_num_instances=2):
    if len(np.unique(label)) >= min_num_instances:
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


def get_valid_slices_per_volume(
    image, gt, fname, save_dir, visualize=False, overwrite_images=True, min_num_instances=2
):
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

        if has_foreground(gt_slice, min_num_instances=min_num_instances):
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


def convert_simple_datasets(image_paths, gt_paths, save_dir, fname_ext, map_to_id=None, extension=".tif"):
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

        image = read_image(image_path, extension=extension)
        gt = read_image(gt_path, extension=extension)

        if has_foreground(gt):
            image = resize_inputs(image)
            gt = resize_inputs(gt, is_label=True)

            if map_to_id is not None:
                assert isinstance(map_to_id, dict)
                for k, v in map_to_id.items():
                    if k in gt:
                        gt[gt == k] = v

            if gt.dtype == "bool":  # for uwaterloo
                gt = gt.astype("uint8")

            imageio.imwrite(trg_image_path, image, compression="zlib")
            imageio.imwrite(trg_gt_path, gt, compression="zlib")


#
# Medical Imaging Datasets
#


def for_sega(save_dir, split_choice):
    """Task: Aorta Segmentation in CT Scans.

    We have three chunks of data: kits, rider, dongyang.
    - for validation: 50*3 (respectively)
    - for testing: ...
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.sega._get_sega_paths(
        path=os.path.join(ROOT, "sega"), data_choice=split_choice, download=False,
    )

    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = read_image(image_path, extension=".nrrd")
        gt = read_image(gt_path, extension=".seg.nrrd")

        image_id = Path(image_path).stem

        # make channels first
        image, gt = image.transpose(2, 0, 1), gt.transpose(2, 0, 1)

        get_valid_slices_per_volume(
            image=image,
            gt=gt,
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
    convert_simple_datasets(
        image_paths=image_paths, gt_paths=gt_paths, save_dir=save_dir, fname_ext=fext, map_to_id={255: 1},
    )
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

    NOTE 1: We choose first 25 patients for extracting the slices.
    NOTE 2: We choose the slices with all 4 cardiac structures present.
    - for validation: 50 * 2
    - for testing: ...
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.camus._get_camus_paths(
        path=os.path.join(ROOT, "camus"), chamber=chamber_choice, download=False,
    )

    # HACK:
    image_paths, gt_paths = image_paths[:25], gt_paths[:25]

    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = read_image(image_path, extension=".nii.gz")
        gt = read_image(gt_path, extension=".nii.gz")

        image_id = Path(image_path).stem

        # make channels first
        image, gt = image.transpose(2, 0, 1), gt.transpose(2, 0, 1)

        get_valid_slices_per_volume(
            image=image,
            gt=gt,
            fname=f"camus_{image_id}",
            save_dir=save_dir,
            min_num_instances=4,
        )

    _get_val_test_splits(save_dir=save_dir, val_fraction=50, fname_ext="camus_")


def for_montgomery(save_dir):
    """Task: Lung Segmentation in Chest X-Rays Images.

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


def for_oimhs(save_dir):
    """Task: Macular region segmentation in OCT images.

    - for validation: 10
    - for testing:
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.oimhs._get_oimhs_paths(
        path=os.path.join(ROOT, "oimhs"), split="test", download=False
    )

    fext = "oimhs_"
    convert_simple_datasets(image_paths=image_paths, gt_paths=gt_paths, save_dir=save_dir, fname_ext=fext)
    _get_val_test_splits(save_dir=save_dir, val_fraction=10, fname_ext=fext)


def for_isic(save_dir):
    """Task: Skin lesion segmentation in dermoscopy images.
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.isic._get_isic_paths(path=os.path.join(ROOT, "isic"), split="test", download=False)

    fext = "isic_"
    convert_simple_datasets(image_paths=image_paths, gt_paths=gt_paths, save_dir=save_dir, fname_ext=fext)
    _get_val_test_splits(save_dir=save_dir, val_fraction=10, fname_ext=fext)


def for_papila(save_dir, task):
    """Task: Optic disc and optic cup segmentation in fundus images
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.papila._get_papila_paths(
        path=os.path.join(ROOT, "papila"), task=task, expert_choice="exp1", download=True,
    )

    fext = "papila_"
    convert_simple_datasets(image_paths=image_paths, gt_paths=gt_paths, save_dir=save_dir, fname_ext=fext)
    _get_val_test_splits(save_dir=save_dir, val_fraction=10, fname_ext=fext)


def for_osic_pulmofib(save_dir):
    """Task: Lung, heart and trachea segmentation in CT scans.
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.osic_pulmofib._get_osic_pulmofib_paths(
        path=os.path.join(ROOT, "osic_pulmofib"), download=True
    )

    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = read_image(image_path, extension=".nii.gz")
        gt = read_image(gt_path, extension=".nii.gz")

        image_id = Path(image_path).stem

        # make channels first
        image, gt = image.transpose(2, 0, 1), gt.transpose(2, 0, 1)

        get_valid_slices_per_volume(
            image=image,
            gt=gt,
            fname=f"osic_pulmofib_{image_id}",
            save_dir=save_dir,
            min_num_instances=4,
        )

    _get_val_test_splits(save_dir=save_dir, val_fraction=50, fname_ext="osic_pulmofib_")


def for_m2caiseg(save_dir):
    """Instrument and organ segmentation in laparoscopy.
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.m2caiseg._get_m2caiseg_paths(
        path=os.path.join(ROOT, "m2caiseg"), split="test", download=True,
    )

    fext = "m2caiseg_"
    convert_simple_datasets(image_paths=image_paths, gt_paths=gt_paths, save_dir=save_dir, fname_ext=fext)
    _get_val_test_splits(save_dir=save_dir, val_fraction=10, fname_ext=fext)


def for_siim_acr(save_dir):
    """Pneumothorax segmentation in X-Ray.
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.siim_acr._get_siim_acr_paths(
        path=os.path.join(ROOT, "siim_acr"), split="test", download=True,
    )

    fext = "siim_acr_"
    convert_simple_datasets(image_paths=image_paths, gt_paths=gt_paths, save_dir=save_dir, fname_ext=fext)
    _get_val_test_splits(save_dir=save_dir, val_fraction=10, fname_ext=fext)


def for_jnu_fim(save_dir):
    """Task: Fetal head and pubic symphysis segmentation in ultrasound.
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.jnuifm._get_jnuifm_paths(path=os.path.join(ROOT, "jnuifm"), download=True)

    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = read_image(image_path, extension=".mha")
        gt = read_image(gt_path, extension=".mha")

        image_id = Path(image_path).stem

        # make channels first
        image, gt = image.transpose(2, 0, 1), gt.transpose(2, 0, 1)

        get_valid_slices_per_volume(
            image=image,
            gt=gt,
            fname=f"jnuifm_{image_id}",
            save_dir=save_dir,
            min_num_instances=4,
        )

    _get_val_test_splits(save_dir=save_dir, val_fraction=10, fname_ext="jnuifm_")


def for_microusp(save_dir):
    """Task: Prostate segmentation in micro-ultrasound.
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.micro_usp._get_micro_usp_paths(
        path=os.path.jon(ROOT, "microusp"), split="test", download=True
    )

    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = read_image(image_path, extension=".mha")
        gt = read_image(gt_path, extension=".mha")

        image_id = Path(image_path).stem

        # make channels first
        image, gt = image.transpose(2, 0, 1), gt.transpose(2, 0, 1)

        get_valid_slices_per_volume(
            image=image,
            gt=gt,
            fname=f"microusp_{image_id}",
            save_dir=save_dir,
            min_num_instances=4,
        )

    _get_val_test_splits(save_dir=save_dir, val_fraction=10, fname_ext="microusp_")


def for_cbis_ddsm(save_dir):
    """Task: (Lesion) Mass segmentation in mammography
    """
    if _check_preprocessing(save_dir=save_dir):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.cbis_ddsm._get_cbis_ddsm_paths(
        path=os.path.join(ROOT, "cbis_ddsm"), split="Test", task="Mass", tumour_type=None, download=True,
    )

    fext = "cbis_ddsm_"
    convert_simple_datasets(image_paths=image_paths, gt_paths=gt_paths, save_dir=save_dir, fname_ext=fext)
    _get_val_test_splits(save_dir=save_dir, val_fraction=10, fname_ext=fext)


def for_btcv(save_dir):
    """Task: Organ segmentation in abdominal CT scans.
    """
    if _check_preprocessing(save_dir=save_dir) and _check_preprocessing(save_dir=f"{save_dir}_3d"):
        print("Looks like the preprocessing has completed.")
        return

    image_paths, gt_paths = medical.btcv._get_raw_and_label_paths(
        path=os.path.join(ROOT, "btcv"), anatomy=["Abdomen"]
    )
    # taking the chosen test split - last 8 volumes
    image_paths, gt_paths = image_paths["Abdomen"][22:], gt_paths["Abdomen"][22:]

    # for this dataset, we store the 2d and 3d slices both

    # directory to save 3d volumes
    image3d_dir = os.path.join(f"{save_dir}_3d", "test", "images")
    gt3d_dir = os.path.join(f"{save_dir}_3d", "test", "ground_truth")
    os.makedirs(image3d_dir, exist_ok=True)
    os.makedirs(gt3d_dir, exist_ok=True)

    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = read_image(image_path, extension=".nii.gz")
        gt = read_image(gt_path, extension=".nii.gz")

        image_id = Path(image_path).stem.split(".")[0]

        # make channels first
        image, gt = image.transpose(2, 0, 1), gt.transpose(2, 0, 1)

        # let's do 2d first
        get_valid_slices_per_volume(
            image=image,
            gt=gt,
            fname=f"btcv_{image_id}",
            save_dir=save_dir,
            min_num_instances=4,
        )

        # next, let's store the 3d volumes as well
        image_nifti = nib.Nifti2Image(image, np.eye(4))
        gt_nifti = nib.Nifti2Image(gt, np.eye(4))

        neu_image_path = os.path.join(image3d_dir, f"{image_id}_0000.nii.gz")
        neu_gt_path = os.path.join(gt3d_dir, f"{image_id}.nii.gz")

        nib.save(image_nifti, neu_image_path)
        nib.save(gt_nifti, neu_gt_path)

    _get_val_test_splits(save_dir=save_dir, val_fraction=10, fname_ext="btcv_")


def for_cholecseg8k(save_dir):
    ...


def for_piccolo(save_dir):
    ...


def for_duke_liver(save_dir):
    ...


def for_toothfairy(save_dir):
    ...


def _preprocess_datasets(save_dir):
    for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "kits"), split_choice="KiTS")
    for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "rider"), split_choice="Rider")
    for_sega(save_dir=os.path.join(save_dir, "sega", "slices", "dongyang"), split_choice="Dongyang")
    for_uwaterloo_skin(save_dir=os.path.join(save_dir, "uwaterloo_skin", "slices"))
    for_idrid(save_dir=os.path.join(save_dir, "idrid", "slices"))
    for_camus(save_dir=os.path.join(save_dir, "camus", "slices", "2ch"), chamber_choice=2)
    for_camus(save_dir=os.path.join(save_dir, "camus", "slices", "4ch"), chamber_choice=4)
    for_montgomery(save_dir=os.path.join(save_dir, "montgomery", "slices"))
    for_oimhs(save_dir=os.path.join(save_dir, "oimhs", "slices"))
    for_btcv(save_dir=os.path.join(save_dir, "btcv", "slices"))
    for_isic(save_dir=os.path.join(save_dir, "isic", "slices"))
    for_papila(save_dir=os.path.join(save_dir, "isic", "slices", "cup"), task="cup")
    for_papila(save_dir=os.path.join(save_dir, "isic", "slices", "disc"), task="disc")
    for_osic_pulmofib(save_dir=os.path.join(save_dir, "osic_pulmofib", "slices"))
    for_m2caiseg(save_dir=os.path.join(save_dir, "m2caiseg", "slices"))
    for_siim_acr(save_dir=os.path.join(save_dir, "siim_acr", "slices"))


def main():
    save_dir = "/scratch/share/cidas/cca/data"
    _preprocess_datasets(save_dir=save_dir)


if __name__ == "__main__":
    main()
