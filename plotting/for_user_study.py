# we choose images from four different modalities: CT, MRI, X-Ray and NBI

import os
from tqdm import tqdm

import random
import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt

from torch_em.data.datasets import medical
from torch_em.transform.raw import normalize

from micro_sam.evaluation.model_comparison import _overlay_mask

from tukra.utils import read_image


ROOT = "/scratch/share/cidas/cca/data"


def _get_image_gt_pairs(dataset_name, n_pairs=1):
    os.makedirs("./figures/images", exist_ok=True)

    if dataset_name == "amos":
        image_paths, gt_paths = medical.amos._get_amos_paths(
            path=os.path.join(ROOT, "amos"), split="val", modality="MRI", download=False
        )
        extension = ".nii.gz"

    elif dataset_name == "jnu-ifm":
        image_paths, gt_paths = medical.jnuifm._get_jnuifm_paths(
            path=os.path.join(ROOT, "jnu-ifm"), download=False
        )
        extension = ".mha"

    elif dataset_name == "montgomery":
        image_paths, gt_paths = medical.montgomery._get_montgomery_paths(
            path=os.path.join(ROOT, "montgomery"), download=False
        )
        extension = ".tif"

    elif dataset_name == "btcv":
        image_paths, gt_paths = medical.btcv._get_raw_and_label_paths(
            path=os.path.join(ROOT, "btcv"), anatomy=["Cervix"]
        )
        image_paths, gt_paths = image_paths["Cervix"], gt_paths["Cervix"]
        extension = ".nii.gz"

    else:
        raise ValueError

    chosen_indices = random.sample(list(range(len(image_paths))), n_pairs)
    chosen_image_paths = [image_paths[i] for i in chosen_indices]
    chosen_gt_paths = [gt_paths[i] for i in chosen_indices]

    for i, (image_path, gt_path) in enumerate(zip(chosen_image_paths, chosen_gt_paths)):
        image = read_image(image_path, extension)
        gt = read_image(gt_path, extension)

        if dataset_name == "jnu-ifm":
            image = image[0]

        if gt.ndim == 3:
            # we choose the middle slice for the image and label
            if dataset_name in ["amos", "btcv"]:
                image, gt = image.transpose(2, 0, 1), gt.transpose(2, 0, 1)

            image = normalize(image) * 255

            if dataset_name in ["amos", "btcv"]:
                chosen_slice = int(image.shape[0] / 2)

            image, gt = image[chosen_slice, ...], gt[chosen_slice, ...]
            image, gt = np.rot90(image, k=3), np.rot90(gt, k=3)

        fig, ax = plt.subplots(1, 3, figsize=(20, 20))
        ax[0].imshow(image.astype("uint8"), cmap="gray")
        ax[0].axis("off")
        ax[0].set_title("Image")

        ax[1].imshow(gt, cmap="gray")
        ax[1].axis("off")
        ax[1].set_title("Ground Truth")

        ax[2].imshow(_overlay_mask(image, gt > 0))
        ax[2].axis("off")
        ax[2].set_title("Overlay")

        plt.tight_layout()
        plt.savefig(f"./figures/{dataset_name}_{i:02}.png", bbox_inches="tight")
        plt.close()

        # let's store the images as well
        imageio.imwrite(f"./figures/images/{dataset_name}_{i:02}_image.tif", image)
        imageio.imwrite(f"./figures/images/{dataset_name}_{i:02}_labels.tif", gt)


def _extract_images():
    dataset_names = ["amos", "jnu-ifm", "montgomery", "btcv"]
    for dname in tqdm(dataset_names):
        _get_image_gt_pairs(dataset_name=dname, n_pairs=10)


def _plot_user_study_samples(dataset_name):

    if dataset_name == "amos":
        dchoice = "05"
        id = 6  # liver
    elif dataset_name == "btcv":
        dchoice = "07"
        id = 1  # bladder
    else:
        raise ValueError

    image_path = f"./figures/images/{dataset_name}_{dchoice}_image.tif"
    gt_path = f"./figures/images/{dataset_name}_{dchoice}_labels.tif"

    image = imageio.imread(image_path)
    gt = imageio.imread(gt_path)

    fig, ax = plt.subplots(1, 3, figsize=(20, 20))
    ax[0].imshow(image.astype("uint8"), cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Image", fontsize=24)

    ax[1].imshow(gt == id, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("Ground Truth", fontsize=24)

    ax[2].imshow(_overlay_mask(image, (gt == id)))
    ax[2].axis("off")
    ax[2].set_title("Overlay", fontsize=24)

    plt.tight_layout()
    plt.savefig(f"./figures/user_study_{dataset_name}.png", bbox_inches="tight")
    plt.close()


def main():
    # _extract_images()

    _plot_user_study_samples("amos")
    _plot_user_study_samples("btcv")


if __name__ == "__main__":
    main()
