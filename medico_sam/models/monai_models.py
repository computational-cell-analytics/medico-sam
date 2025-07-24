from typing import Tuple

import torch.nn as nn

from monai.networks.nets import SwinUNETR


def get_monai_models(
    image_size: Tuple[int, ...],
    in_channels: int = 1,
    out_channels: int = 1,
    ndim: int = 2,
) -> nn.Module:
    """Get open-source networks for semantic segmentation from MONAI framework.

    Args:
        image_size: The spatial dimensions of input image arrays.
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        ndim: The number of dimensions.

    Returns:
        The SwinUNETR model.
    """
    model = SwinUNETR(
        img_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=48,
        spatial_dims=ndim,  # Defines the architecture's input type.
        use_checkpoint=True,
    )

    return model
