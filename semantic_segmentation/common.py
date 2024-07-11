import numpy as np
from math import ceil, floor
from torch_em.transform.raw import normalize


class LabelTrafoToBinary:
    def __call__(self, labels):
        labels = (labels > 0).astype(labels.dtype)
        return labels


class RawTrafoFor3dInputs:
    def _normalize_inputs(self, raw):
        raw = normalize(raw)
        raw = raw * 255
        return raw

    def _set_channels_for_inputs(self, raw):
        raw = np.stack([raw] * 3, axis=0)
        return raw

    def __call__(self, raw):
        raw = self._normalize_inputs(raw)
        raw = self._set_channels_for_inputs(raw)
        return raw


# for sega
class RawResizeTrafoFor3dInputs(RawTrafoFor3dInputs):
    def __init__(self, desired_shape, padding="constant"):
        super().__init__()
        self.desired_shape = desired_shape
        self.padding = padding

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

        return raw


# for sega
class LabelResizeTrafoFor3dInputs:
    def __init__(self, desired_shape, padding="constant"):
        self.desired_shape = desired_shape
        self.padding = padding

    def __call__(self, labels):
        # binarize the samples
        labels = (labels > 0).astype("float32")

        # let's pad the labels
        tmp_ddim = (
           self.desired_shape[0] - labels.shape[0],
           self.desired_shape[1] - labels.shape[1],
           self.desired_shape[2] - labels.shape[2]
        )
        ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2, tmp_ddim[2] / 2)
        labels = np.pad(
            labels,
            pad_width=(
                (ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1])), (ceil(ddim[2]), floor(ddim[2]))
            ),
            mode=self.padding
        )

        return labels