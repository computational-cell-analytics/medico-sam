import os
import sys

from torch_em.data.datasets import medical
from torch_em.transform.raw import normalize

from tukra.io import read_image


def _load_raw_and_label_volumes(raw_path, label_path, ensure_8bit=True, channels_first=True):
    raw = read_image(raw_path)
    label = read_image(label_path)

    if ensure_8bit:
        # Ensure inputs are 8 bit.
        if raw.max() > 255:
            raw = normalize(raw) * 255

    if channels_first:  # Ensure volumes are channels first.
        raw, label = raw.transpose(2, 0, 1), label.transpose(2, 0, 1)

    # Ensure labels are integers.
    label = label.astype("uint32")

    assert raw.shape == label.shape

    return raw, label


def _get_data_paths(path, dataset_name):
    sys.path.append("..")
    from util import SEMANTIC_CLASS_MAPS

    # Get paths to the volumetric data.
    path_to_volumes = {
        "sega": medical.sega.get_sega_paths(path=os.path.join(path, "sega"), data_choice="Dongyang"),
    }

    assert dataset_name in path_to_volumes.keys(), f"'{dataset_name}' is not a supported dataset."

    raw_paths, label_paths = path_to_volumes[dataset_name]
    semantic_maps = SEMANTIC_CLASS_MAPS[dataset_name]

    return raw_paths, label_paths, semantic_maps
