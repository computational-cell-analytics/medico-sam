import numpy as np
from math import ceil, floor

from torch_em.transform.raw import normalize, normalize_percentile


# TODO: In future, combine all raw transforms into one (?)
class RawTrafnsformJointTraining:
    def __init__(self, modality: str = "CT"):
        self.modality = modality

    def __call__(self, raw: np.ndarray):
        if self.modality == "CT":
            raw = normalize_percentile(raw)  # Percentile normalization.
            raw = raw * 255  # Convert to 8-bit.
        else:
            raise NotImplementedError

        return raw


class RawTrafoFor3dInputs:
    def __init__(self, switch_last_axes=False):
        self.switch_last_axes = switch_last_axes

    def _normalize_inputs(self, raw):
        raw = normalize(raw)
        raw = raw * 255
        return raw

    def _set_channels_for_inputs(self, raw):
        raw = np.stack([raw] * 3, axis=0)
        return raw

    def _switch_last_axes_for_inputs(self, raw):
        raw = raw.transpose(0, 1, 3, 2)
        return raw

    def __call__(self, raw):
        raw = self._normalize_inputs(raw)
        raw = self._set_channels_for_inputs(raw)
        if self.switch_last_axes:
            raw = self._switch_last_axes_for_inputs(raw)
        return raw


# for 3d volumes like SegA
class RawResizeTrafoFor3dInputs(RawTrafoFor3dInputs):
    def __init__(self, desired_shape, padding="constant", switch_last_axes=False):
        super().__init__()
        self.desired_shape = desired_shape
        self.padding = padding
        self.switch_last_axes = switch_last_axes

    def __call__(self, raw):
        raw = self._normalize_inputs(raw)

        # let's pad the inputs
        tmp_ddim = (
           self.desired_shape[0] - raw.shape[0],
           self.desired_shape[1] - raw.shape[1],
           self.desired_shape[2] - raw.shape[2]
        )
        ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2, tmp_ddim[2] / 2)
        raw = np.pad(
            raw,
            pad_width=(
                (ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1])), (ceil(ddim[2]), floor(ddim[2]))
            ),
            mode=self.padding
        )

        raw = self._set_channels_for_inputs(raw)

        if self.switch_last_axes:
            raw = self._switch_last_axes_for_inputs(raw)

        return raw
