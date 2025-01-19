import os
from glob import glob

import h5py

import napari


ROOT = "/home/anwai/data/medico-sam/interactive_seg_3d"


def main():
    for dname in ["osic_pulmofib", "microusp", "lgg_mri", "duke_liver"]:
        res_path = glob(os.path.join(ROOT, f"{dname}*.h5"))[-1]

        with h5py.File(res_path, "r") as f:
            # Get the inputs
            raw = f["raw"][:]

            # Get the segmentation
            seg = f["segmentation"][:]

        # Extract mid-slice
        raw_slice = raw.copy()
        z = len(raw_slice) // 2
        raw_slice[:z] = 0
        raw_slice[(z+2):] = 0

        v = napari.Viewer()
        v.axes.visible = True
        v.add_image(raw_slice, name="Raw")
        v.add_labels(seg, name="Segmentation")

        napari.run()


if __name__ == "__main__":
    main()
