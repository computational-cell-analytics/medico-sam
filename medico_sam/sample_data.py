r"""Sample medical imaging data.

You can change the download location for the sample data and model weights
by setting the environment variable: MEDICOSAM_CACHEDIR

By default sample data is downloaded to a folder named 'medico_sam/sample_data'.
inside your default cache directory, eg:
    * Mac: ~/Library/Caches/<AppName>
    * Unix: ~/.cache/<AppName> or the value of the XDG_CACHE_HOME environment variable, if defined.
    * Windows: C:\Users\<user>\AppData\Local\<AppAuthor>\<AppName>\Cache
"""

import os
from typing import Union
from pathlib import Path

import pooch

from torch_em.data.datasets.util import unzip


def fetch_dermoscopy_example_data(save_directory: Union[str, os.PathLike]) -> str:
    """Download the sample images for the 2d annotator.

    Args:
        save_directory: Root folder to save the downloaded data.

    Returns:
        The folder that contains the downloaded data.
    """
    # micro-sam currently supports only microscopy images under sample dataset.
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    fname = "uwaterloo_skin_sample_image.jpg"
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/8CPxNCTtysJOSbP/download",
        known_hash="73ddbb86bac87847d653865673793c28c96b3fca6048f29e059089b80d4fbb17",
        fname=fname,
        path=save_directory,
        progressbar=True,
    )
    return os.path.join(save_directory, fname)


def fetch_ct_example_data(save_directory: Union[str, os.PathLike]) -> str:
    """Download the sample images for the 3d annotator.

    Args:
        save_directory: Root folder to save the downloaded data.

    Returns:
        The folder that contains the downloaded data.
    """
    # micro-sam currently supports only microscopy images under sample dataset.
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    fname = "osic_pulmofib_sample_image.nii"
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/oFLvfMR5ppczxFh/download",
        known_hash="32a21b6001ee0e1f5d5f336a472cde38a93e7dfb8f1c63ca31e215a36eb7f90d",
        fname=fname,
        path=save_directory,
        progressbar=True,
    )
    return os.path.join(save_directory, fname)


def fetch_fundus_example_data(save_directory: Union[str, os.PathLike]) -> str:
    """Download the sample images for the image series annotator.

    Args:
        save_directory: Root folder to save the downloaded data.

    Returns:
        The folder that contains the downloaded data.
    """
    # micro-sam currently supports only microscopy images under sample dataset.
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    fname = "papila.zip"
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/k66IZo3pU7TaSaQ/download",
        known_hash="b1c1093669451c160eddcbadc25c8c184b9e96952cd5c7c9e8ea3310eb24fbc7",
        fname=fname,
        path=save_directory,
        progressbar=True,
    )
    unzip(
        zip_path=os.path.join(save_directory, fname),
        dst=os.path.join(save_directory, f"{fname}.unzip"),
        remove=False
    )
    data_folder = os.path.join(save_directory, f"{fname}.unzip")
    assert os.path.exists(data_folder), data_folder
    return data_folder
