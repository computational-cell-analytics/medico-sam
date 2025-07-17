from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_em.model.unet import Decoder, ConvBlock2d, Upsampler2d
from torch_em.model.unetr import Deconv2DBlock


class SimpleUNETR3D(nn.Module):
    """Simple design for getting spatial context using 3d convolutions on top of pretrained 2d UNETR decoder.

    Args:
        encoder: The image encoder, adapted for 3d inputs.
        out_channels: The total number of output classes.
        fusion_channels: The depth of 3d fusion block taking pretrained 2d features as inputs.
        final_activation: The final activation block.
    """

    def __init__(
        self,
        encoder: nn.Module,
        out_channels: int = 1,
        fusion_channels: int = 64,
        final_activation: Optional[Union[str, nn.Module]] = None,
    ) -> None:
        super().__init__()

        # Get the image encoder: wrapped 3d ViT.
        self.encoder = encoder

        # Additional parameters
        embed_dim = self.encoder.image_encoder.neck[2].out_channels

        # Create a simple image decoder.
        # First, get the important parameters for the decoder.
        depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]
        self.out_channels = out_channels

        self.decoder = Decoder(
            features=features_decoder,
            scale_factors=scale_factors[::-1],
            conv_block_impl=ConvBlock2d,
            sampler_impl=Upsampler2d,  # uses bilinear interpolation for upsampling
        )

        # TODO: Make the setup modular for allowing skip connections?
        self.deconv1 = Deconv2DBlock(
            in_channels=embed_dim, out_channels=features_decoder[0], use_conv_transpose=False,
        )
        self.deconv2 = Deconv2DBlock(
            in_channels=features_decoder[0], out_channels=features_decoder[1], use_conv_transpose=False,
        )
        self.deconv3 = Deconv2DBlock(
            in_channels=features_decoder[1], out_channels=features_decoder[2], use_conv_transpose=False,
        )
        self.deconv4 = Deconv2DBlock(
            in_channels=features_decoder[2], out_channels=features_decoder[3], use_conv_transpose=False,
        )

        # And further conjunction blocks.
        self.base = ConvBlock2d(embed_dim, features_decoder[0])
        self.deconv_out = Upsampler2d(
            scale_factor=2, in_channels=features_decoder[-1], out_channels=features_decoder[-1]
        )
        self.decoder_head = ConvBlock2d(2 * features_decoder[-1], features_decoder[-1])

        # 3d fusion block.
        self.fusion_3d = nn.Sequential(
            nn.Conv3d(features_decoder[-1], fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(fusion_channels, fusion_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(fusion_channels),
            nn.ReLU(inplace=True),
        )

        # And final classification head.
        self.classifier = nn.Conv3d(fusion_channels, out_channels, 1)
        self.final_activation = self._get_activation(final_activation)

    def _get_activation(self, activation):
        return_activation = None
        if activation is None:
            return None
        if isinstance(activation, nn.Module):
            return activation
        if isinstance(activation, str):
            return_activation = getattr(nn, activation, None)

        if return_activation is None:
            raise ValueError(f"Invalid activation: {activation}")

        return return_activation()

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        scale = long_side_length / max(oldh, oldw)
        newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
        return newh, neww

    def resize_longest_side(self, image: torch.Tensor) -> torch.Tensor:
        target_size = self.get_preprocess_shape(image.shape[3], image.shape[4], self.encoder.img_size)

        target_image = torch.stack(
            [F.interpolate(im, target_size, mode="bilinear", align_corners=False, antialias=True) for im in image],
            dim=0,
        )

        return target_image

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """@private
        """
        device = x.device

        # Normalize the input to SAM statistics
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1, 1).to(device)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1, 1).to(device)

        # Resize the inputs.
        x = self.resize_longest_side(x)
        input_shape = x.shape[-2:]

        x = (x - pixel_mean) / pixel_std
        pad_h = self.encoder.img_size - x.shape[-2]
        pad_w = self.encoder.img_size - x.shape[-1]
        x = F.pad(x, (0, pad_w, 0, pad_h))

        return x, input_shape

    def postprocess_masks(
        self, masks: torch.Tensor, input_size: Tuple[int, int], original_size: Tuple[int, int],
    ) -> torch.Tensor:
        masks = torch.stack(
            [
                F.interpolate(
                    mask, (self.encoder.img_size, self.encoder.img_size), mode="bilinear", align_corners=False
                ) for mask in masks
            ], dim=0,
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = torch.stack(
            [F.interpolate(mask, original_size, mode="bilinear", align_corners=False) for mask in masks],
            dim=0,
        )
        return masks

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the 3d UNETR to the input data.

        Args:
            x: The input tensor.
            kwargs: Additional parameters provided, implemented to allow catching extra arguments.

        Returns:
            The 3d UNETR output.
        """
        # Get original shape information.
        B, C, D, H, W = x.shape
        original_shape = (H, W)

        # Reshape the inputs to the shape expected by the encoder
        # and normalize the inputs if normalization is part of the model.
        x, input_shape = self.preprocess(x)

        # Following the Sam3DWrapper heuristics, transpose the axes so that depth axis is first and channel
        # is second (expected by the adapted transformer)
        x = x.transpose(1, 2)
        assert x.shape[1] == D
        x = x.contiguous().view(-1, C, H, W)  # Performs B*Z to work with 2d convolutions!

        # Run the image encoder
        encoder_outputs = self.encoder(x, d_size=D)  # 'd_size' corresponds to the z-dimension.
        z12 = encoder_outputs

        # And the decoder part
        z9 = self.deconv1(z12)
        z6 = self.deconv2(z9)
        z3 = self.deconv3(z6)
        z0 = self.deconv4(z3)

        updated_from_encoder = [z9, z6, z3]

        x = self.base(z12)
        x = self.decoder(x, encoder_inputs=updated_from_encoder)
        x = self.deconv_out(x)

        x = torch.cat([x, z0], dim=1)
        x = self.decoder_head(x)

        # 3d fusion block
        x = x.view(B, D, -1, *x.shape[-2:]).permute(0, 2, 1, 3, 4)
        x = self.fusion_3d(x)

        # And the classification head.
        x = self.classifier(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        x = self.postprocess_masks(x, input_shape, original_shape)
        return x
