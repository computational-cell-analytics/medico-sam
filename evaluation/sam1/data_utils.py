import os
import sys

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

    if channels_first:  # Ensure volumes are channels first.
        if dataset_name == "sega":  # match the expected axes alignment
            raw, label = raw.transpose(2, 1, 0), label.transpose(2, 1, 0)
        else:
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
        "sega": lambda: medical.sega.get_sega_paths(
            path=os.path.join(path, "sega"), data_choice="Dongyang", download=True
        ),
        "curvas": lambda: medical.curvas.get_curvas_paths(path=os.path.join(path, "curvas"), split="test"),
        "ct_cadaiver": lambda: medical.ct_cadaiver.get_ct_cadaiver_paths(path=os.path.join(path, "ct_cadaiver")),
        "lgg_mri": lambda: medical.lgg_mri.get_lgg_mri_paths(
            path=os.path.join(path, "lgg_mri"), split="test", download=True
        ),
        "duke_liver": lambda: medical.duke_liver.get_duke_liver_paths(
            path=os.path.join(path, "duke_liver"), split="test",
        ),
        "microusp": lambda: medical.micro_usp.get_micro_usp_paths(
            path=os.path.join(path, "micro_usp"), split="test", download=True,
        ),
    }

    assert dataset_name in path_to_volumes.keys(), f"'{dataset_name}' is not a supported dataset."

    input_paths = path_to_volumes[dataset_name]()
    if isinstance(input_paths, tuple):
        raw_paths, label_paths = input_paths
    else:
        raw_paths = label_paths = input_paths

    semantic_maps = SEMANTIC_CLASS_MAPS[dataset_name]

    ensure_channels_first = True
    if dataset_name in ["ct_cadaiver", "lgg_mri"]:
        ensure_channels_first = False

    keys = None
    if dataset_name == "lgg_mri":
        keys = ("raw/flair", "labels")

    return raw_paths, label_paths, semantic_maps, keys, ensure_channels_first
