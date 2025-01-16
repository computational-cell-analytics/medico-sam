import os
import sys

import numpy as np
from skimage.measure import label as connected_components

from torch_em.data.datasets import medical
from torch_em.transform.raw import normalize

from tukra.io import read_image


def _load_raw_and_label_volumes(raw_path, label_path, dataset_name, ensure_8bit=True, channels_first=True, keys=None):
    raw = read_image(raw_path, key=keys if keys is None else keys[0])
    label = read_image(label_path, key=keys if keys is None else keys[1])

    if ensure_8bit:
        # Ensure inputs are 8 bit.
        if raw.max() > 255:
            raw = normalize(raw) * 255
        raw = raw.astype("uint8")

    # Ensure volumes are channels first.
    if channels_first:
        raw, label = raw.transpose(2, 0, 1), label.transpose(2, 0, 1)

    if dataset_name == "segthy":
        raw, label = raw.transpose(0, 2, 1), label.transpose(0, 2, 1)

    # Ensure labels are integers.
    label = np.round(label).astype("uint32")

    # Ensure unique ids for specific class
    if dataset_name in ["kits", "segthy"]:
        label = connected_components(label).astype(label.dtype)

    elif dataset_name == "osic_pulmofib":
        lung_label = (label == 2)
        lung_label = connected_components(lung_label).astype(label.dtype)

        # Change the mapping to work with two lung ids.
        label[label == 2] = 0  # Remove the lung labels
        label[label == 3] = 2  # Move 'trachea' to label id '2'.

        # Now insert both lung labels.
        fids = np.unique(lung_label)[1:]
        for id in fids:  # Add offset to new id
            offset_id = 2 + id
            assert offset_id not in label
            label[lung_label == id] = offset_id

    assert raw.shape == label.shape

    return raw, label


def _get_data_paths(path, dataset_name):
    sys.path.append("..")
    from util import SEMANTIC_CLASS_MAPS

    # Get paths to the volumetric data.
    path_to_volumes = {
        # glioma segmentation in MRI.
        "lgg_mri": lambda: medical.lgg_mri.get_lgg_mri_paths(
            path=os.path.join(path, "lgg_mri"), split="test", download=True
        ),
        # liver segmentation in MRI.
        "duke_liver": lambda: medical.duke_liver.get_duke_liver_paths(
            path=os.path.join(path, "duke_liver"), split="test",
        ),
        # prostate segmentation in MicroUS.
        "microusp": lambda: medical.micro_usp.get_micro_usp_paths(
            path=os.path.join(path, "micro_usp"), split="test", download=True,
        ),
        # thyroid segmentation in US.
        "segthy": lambda: medical.segthy.get_segthy_paths(
            path=os.path.join(path, "segthy"), split="test", source="US", region="thyroid", download=True,
        ),
        # thoracic organ segmentation in CT.
        "osic_pulmofib": lambda: medical.osic_pulmofib.get_osic_pulmofib_paths(
            path=os.path.join(path, "osic_pulmofib"), split="test", download=True,
        ),
        # kidney tumor segmentation in CT.
        "kits": lambda: medical.kits.get_kits_paths(
            path=os.path.join(path, "kits"), download=True,
        ),
    }

    assert dataset_name in path_to_volumes.keys(), f"'{dataset_name}' is not a supported dataset."

    input_paths = path_to_volumes[dataset_name]()
    if isinstance(input_paths, tuple):
        raw_paths, label_paths = input_paths
    else:
        raw_paths = label_paths = input_paths

    # NOTE: We limit the number of images for KiTS
    if dataset_name == "kits":
        raw_paths, label_paths = raw_paths[:20], label_paths[:20]  # evaluate on 20 volumes only.

    semantic_maps = SEMANTIC_CLASS_MAPS[dataset_name + ("_3d" if dataset_name == "osic_pulmofib" else "")]

    ensure_channels_first = True
    # The datasets below already have channels first. We do not need to take care of them.
    if dataset_name in ["lgg_mri", "leg_3d_us", "kits"]:
        ensure_channels_first = False

    keys = None
    # The datasets below are in container format. We need their hierarchy names.
    if dataset_name == "lgg_mri":
        keys = ("raw/flair", "labels")
    elif dataset_name == "kits":
        keys = ("raw", "labels/tumor/rater_1")

    return raw_paths, label_paths, semantic_maps, keys, ensure_channels_first
