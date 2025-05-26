"""Helper function for downloading Segment Anything models.
"""

import os
from pathlib import Path
from typing import Union, Optional

import pooch

import torch
from torch.optim.lr_scheduler import _LRScheduler

from segment_anything import SamPredictor


# this is the default model used in medico_sam
# currently set to the default 'vit_b'
_DEFAULT_MODEL = "vit_b"


#
# Functionality for model download.
# Inspired by: https://github.com/computational-cell-analytics/micro-sam/blob/master/micro_sam/util.py
#


def get_cache_directory() -> None:
    """Get medico-sam cache directory location.

    Users can set the MEDICOSAM_CACHEDIR environment variable for a custom cache directory.
    """
    default_cache_directory = os.path.expanduser(pooch.os_cache("medico_sam"))
    cache_directory = Path(os.environ.get("MEDICOSAM_CACHEDIR", default_cache_directory))
    return cache_directory


def medico_sam_cachedir() -> None:
    """Return the medico-sam cache directory.

    Returns the top level cache directory for medico-sam models and sample data.

    Every time this function is called, we check for any user updates made to
    the MEDICOSAM_CACHEDIR os environment variable since the last time.
    """
    cache_directory = os.environ.get("MEDICOSAM_CACHEDIR") or pooch.os_cache("medico_sam")
    return cache_directory


def models():
    """Return the segmentation models registry.

    We recreate the model registry every time this function is called,
    so any user changes to the default medico-sam cache directory location
    are respected.
    """

    # We use xxhash to compute the hash of the models, see
    # https://github.com/computational-cell-analytics/micro-sam/issues/283
    # (It is now a dependency, so we don't provide the sha256 fallback anymore.)
    # To generate the xxh128 hash:
    #     xxh128sum filename
    registry = {
        # The default segment anything models:
        "vit_b": "xxh128:6923c33df3637b6a922d7682bfc9a86b",
        "vit_l": "xxh128:a82beb3c660661e3dd38d999cc860e9a",
        "vit_h": "xxh128:97698fac30bd929c2e6d8d8cc15933c2",
        # The MedicoSAM models:
        "vit_b_medical_imaging": "xxh128:5be672f1458263a9edc9fd40d7f56ac1",
    }

    urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        # The MedicoSAM models:
        "vit_b_medical_imaging": "https://owncloud.gwdg.de/index.php/s/AB69HGhj8wuozXQ/download",
    }

    models = pooch.create(
        path=os.path.join(medico_sam_cachedir(), "models"), base_url="", registry=registry, urls=urls,
    )
    return models


def get_medico_sam_model(
    model_type: str = _DEFAULT_MODEL,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    use_sam_med2d: bool = False,
    use_sam3d: bool = False,
    encoder_adapter: bool = False,
    **kwargs
) -> SamPredictor:
    """Get the Segment Anything Predictor.

    Args:
        model_type: The Segment Anything model to use. Will use the standard `vit_b` model by default.
            To get a list of all available model names, you can call `get_model_names`.
        device: The device for the model. If `None` is given, it will use GPU if available.
        checkpoint_path: The path to a file with weights that should be used instead of using the
            weights corresponding to the weight file. e.g. if you use weights for SAM with `vit_b` encoder,
            then `model_type` must be given as "vit_b".
        use_sam_med2d: Whether to use SAM-Med2d model for initializing the checkpoints.
        use_sam3d: Whether to use SAM-3d model for initializing the checkpoints.
        encoder_adapter: Whether the encuder in SAM-Med2d has adapter modules.
        kwargs: Additional arguments for the suitable 'get_*_model' function.

    Returns:
        The segment anything predictor.
    """
    # checkpoint_path has not been passed, we download a known model and derive the correct
    # URL from the model_type. If the model_type is invalid pooch will raise an error.
    if checkpoint_path is None:
        model_registry = models()
        checkpoint_path = model_registry.fetch(model_type, progressbar=True)

    model_kwargs = {
        "model_type": model_type, "device": device, "checkpoint_path": checkpoint_path,
    }

    assert (use_sam_med2d + use_sam3d) < 2, "Please use either of 'use_sam_med2d' or 'use_sam3d'."

    if use_sam_med2d:
        from medico_sam.models.sam_med2d.util import get_sam_med2d_model
        _fetch_model = get_sam_med2d_model
        model_kwargs["encoder_adapter"] = encoder_adapter

    else:
        if use_sam3d:
            from micro_sam.models.sam_3d_wrapper import get_sam_3d_model
            _fetch_model = get_sam_3d_model

        else:
            from micro_sam.util import get_sam_model
            _fetch_model = get_sam_model

        model_kwargs = {**model_kwargs, **kwargs}

    return _fetch_model(**model_kwargs)


#
# learning rate scheduler using warnup
#


class LinearWarmUpScheduler(_LRScheduler):
    """Wrapper for custom learning rate scheduler that applied linear warmup,
    followed by a primary scheduler (eg. ReduceLROnPlateau) after the warmup.

    Args:
        optimizer: The optimizer
        warmup_epochs (int): Equivalent to the number of epochs for linear warmup.
        main_scheduler: The scheduler.
        last_epoch (int): The index of the last epoch.
    """
    def __init__(self, optimizer, warmup_epochs, main_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.main_scheduler = main_scheduler
        self.is_warmup_finished = False

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            self.is_warmup_finished = True
            return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics=None, epoch=None):
        if self.is_warmup_finished:
            self.main_scheduler.step(metrics, epoch)
        else:
            super().step()

    def _get_closed_form_lr(self):
        return self.get_lr()
