import os

import numpy as np
import matplotlib.pyplot as plt

from torch_em.data.datasets import medical
from torch_em.transform.generic import ResizeLongestSideInputs

from micro_sam.evaluation.model_comparison import _overlay_outline

from tukra.io import read_image


ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def for_fig_1a():
    # Get a cummulative image for multiple datasets.
    get_paths = {
        # MRI
        "chaos": lambda: medical.chaos.get_chaos_paths(path=os.path.join(ROOT, "chaos"), split="train", modality="MRI"),
        # US data
        "segthy": lambda: medical.segthy.get_segthy_paths(path=os.path.join(ROOT, "segthy"), split="test", source="US"),
        # CT data
        "osic_pulmofib": lambda: medical.osic_pulmofib.get_osic_pulmofib_paths(
            path=os.path.join(ROOT, "osic_pulmofib"), split="test",
        ),
        # X-Ray
        "montgomery": lambda: medical.montgomery.get_montgomery_paths(path=os.path.join(ROOT, "montgomery")),
        # Fundus
        "papila": lambda: medical.papila.get_papila_paths(path=os.path.join(ROOT, "papila"), split="test"),
        # OCT
        "oimhs": lambda: medical.oimhs.get_oimhs_paths(path=os.path.join(ROOT, "oimhs"), split="test"),
        # Dermoscopy
        "uwaterloo_skin": lambda: medical.uwaterloo_skin._get_uwaterloo_skin_paths(
            path=os.path.join(ROOT, "uwaterloo_skin"), download=False,
        ),
        # Endoscopy
        "m2caiseg": lambda: medical.m2caiseg._get_m2caiseg_paths(
            path=os.path.join(ROOT, "m2caiseg"), split="test", download=False
        ),
        # Mammography
        "cbis_ddsm": lambda: medical.cbis_ddsm.get_cbis_ddsm_paths(
            path=os.path.join(ROOT, "cbis_ddsm"), split="Test", task="Mass", ignore_mismatching_pairs=True,
        )
    }

    fig, ax = plt.subplots(3, 3, figsize=(30, 30))
    ax = ax.flatten()

    for i, dname in enumerate(get_paths.keys()):
        input_paths = get_paths[dname]()
        if isinstance(input_paths, tuple):
            image_paths, gt_paths = input_paths
        else:
            image_paths = gt_paths = input_paths

        id = 0
        if dname in ["oimhs"]:
            id = 10
        elif dname == "cbis_ddsm":
            id = -10

        image = read_image(image_paths[id])
        gt = read_image(gt_paths[id])

        if dname in ["chaos", "segthy", "osic_pulmofib"]:
            image, gt = image.transpose(2, 0, 1), gt.transpose(2, 0, 1)

        # Normalize images.
        from torch_em.transform.raw import normalize
        image = normalize(image) * 255
        image = image.astype(int)

        # let's search for valid labels
        if gt.ndim == 3:  # For 3d, we start checking from the mid-slice.
            mid_slice = int(gt.shape[0] / 2)

            if dname == "chaos":
                mid_slice += 8

            if dname == "osic_pulmofib":
                mid_slice -= 6

            image = image[mid_slice, ...]
            gt = gt[mid_slice, ...]

            if dname == "segthy":  # align axes
                image, gt = np.rot90(image, k=1), np.rot90(gt, k=1)

        # Convert all images to RGB
        if not image.ndim == 3:
            image = np.stack([image] * 3, axis=-1)

        # Transform images.
        raw_trafo = ResizeLongestSideInputs(target_shape=(1024, 1024), is_rgb=True)
        label_trafo = ResizeLongestSideInputs(target_shape=(1024, 1024), is_label=True)

        # Let's make channels first for the raw trafo
        # We do all the channel logic stuff to ensure channels last at the end.
        image = raw_trafo(image.transpose(2, 0, 1)).transpose(1, 2, 0)
        gt = label_trafo(gt)

        # Finally, plot them into one place.
        image = _overlay_outline(image, gt, outline_dilation=2)
        ax[i].imshow(image, cmap="gray")
        ax[i].axis("off")

    plt.subplots_adjust(hspace=0.01, wspace=0.01)
    plt.savefig("./fig_1a_medical_dataset_images.png", bbox_inches="tight")
    plt.savefig("./fig_1a_medical_dataset_images.svg", bbox_inches="tight")


def main():
    for_fig_1a()


if __name__ == "__main__":
    main()
