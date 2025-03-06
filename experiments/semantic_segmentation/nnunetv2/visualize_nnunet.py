"""These scripts are for visualizing nnUNet predictions for Alyona's work!
"""

import os
from glob import glob
from pathlib import Path

import numpy as np
import nibabel as nib
import imageio.v3 as imageio

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def check_predictions(dataset_name):

    prob_paths = sorted(glob("/media/anwai/ANWAI/norm_experiments/nnunet_predictions/*.npz"))
    data_dir = f"/media/anwai/ANWAI/data/{dataset_name}"

    if len(prob_paths) == 0:
        print(f"No probabilities found for '{dataset_name}'. We can't debug this further.")
        return

    for i, fpath in enumerate(prob_paths):

        # HACK: the second image has a better visual to show.
        if i == 0:
            continue

        # Get the inputs and visualize them
        image = nib.load(os.path.join(data_dir, "imagesTs", f"{Path(fpath).stem}_0000.nii.gz")).get_fdata()
        gt = nib.load(os.path.join(data_dir, "labelsTs", f"{Path(fpath).stem}.nii.gz")).get_fdata()

        # Match image and labels to probability maps' axes
        image = image.transpose(2, 1, 0)
        gt = gt.transpose(2, 1, 0)

        # Get the probabilities
        # HACK: store this locally to test stuff faster
        tmp_path = "test.tif"
        if os.path.exists(tmp_path):
            prob = imageio.imread(tmp_path)
        else:
            prob = np.load(fpath)["probabilities"]
            prob = prob[1]  # Check one channel only
            imageio.imwrite(tmp_path, prob, compression="zlib")

        # Downscale all inputs to fit into memory for visualizing via napari.
        # from skimage.transform import rescale
        # image = rescale(image, scale=0.5, anti_aliasing=True, preserve_range=True).astype(image.dtype)
        # gt = rescale(gt, scale=0.5, order=0, anti_aliasing=False, preserve_range=True).astype("uint8")
        # prob = rescale(prob, scale=0.5, anti_aliasing=True, preserve_range=True).astype(prob.dtype)

        # HACK: Get the desired slice
        z_slice = 376
        _prob = prob[z_slice]

        # HACK: Get a crop for the slice which fits the proper tile grids.
        _prob = _prob[:576, :1536]

        blue_cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue", [(0, "black"), (1, "dodgerblue")])
        vmin, vmax = 0, 0.00025
        plt.imshow(_prob, cmap=blue_cmap, vmin=vmin, vmax=vmax)

        def _get_additional_params():
            # Get the ticks as expected
            tile_shape = (192, 256)
            yticks = np.arange(0, _prob.shape[0], tile_shape[0])
            xticks = np.arange(0, _prob.shape[1], tile_shape[1])

            yticks = yticks[yticks != 0]
            xticks = xticks[xticks != 0]

            # Set the desired ticks
            plt.yticks(yticks, [])
            plt.xticks(xticks, [])

            # Get the tick marks on all edges.
            plt.tick_params(
                axis='both', direction='in', color='red', length=17.5,
                width=1.5, top=True, bottom=True, left=True, right=True
            )

            # Get a border around the image.
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

        _get_additional_params()

        plt.savefig("./prob_plot.svg", bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        # Next, get the plots for image overlayed with labels.
        _image, _gt = image[z_slice], gt[z_slice]
        _image, _gt = _image[:576, :1536], _gt[:576, :1536]

        # Get the boundaries
        from scipy.ndimage import binary_dilation
        from skimage.segmentation import find_boundaries
        _gt = find_boundaries(_gt == 1).astype(_gt.dtype)
        _gt = binary_dilation(_gt, iterations=6)

        # Create overlay
        _image = _image.astype(np.float32) / _image.max()
        _image_rgb = np.stack([_image] * 3, axis=-1)
        overlayed_image = _image_rgb.copy()
        overlayed_image[_gt == 1] = np.array([0.12, 0.56, 1.0])

        plt.imshow(overlayed_image)
        _get_additional_params()
        plt.savefig("./inputs_plot.svg", bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        break

        # Visualize the probabilities
        # import napari
        # v = napari.Viewer()
        # v.add_image(image)
        # v.add_labels(gt)
        # v.add_image(prob)
        # napari.run()


if __name__ == "__main__":
    check_predictions("leg_3d_us")
