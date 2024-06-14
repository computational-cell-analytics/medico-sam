import os
from typing import Optional, Union

import torch

from .predictor_sam_med2d import SamPredictor
from .build_sam_sam_med2d import sam_model_registry

from micro_sam.util import get_device


DEFAULT_CHECKPOINT = ""


class SAM_Med2d_Args:
    def __init__(
        self,
        image_size: int,
        sam_checkpoint: Union[os.PathLike, str],
        encoder_adapter: bool,
    ):
        self.image_size = image_size
        self.sam_checkpoint = sam_checkpoint
        self.encoder_adapter = encoder_adapter


def get_sam_med2d_model(
    model_type: str = "vit_b",
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    encoder_adapter: bool = False,
):
    device = get_device(device)

    assert model_type is not None

    # Loading the SAM-Med2d models
    state = torch.load(checkpoint_path, map_location=device)
    if "optimizer" in state.keys():
        model_state = state["model"]
    else:
        model_state = state

    args = SAM_Med2d_Args(image_size=256, sam_checkpoint=checkpoint_path, encoder_adapter=encoder_adapter)

    sam = sam_model_registry[model_type](args=args)
    sam.load_state_dict(model_state)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    return predictor
