import numpy as np
from math import ceil, floor


class LabelTrafoToBinary:
    def _binarise_labels(self, labels):
        labels = (labels > 0).astype(labels.dtype)
        return labels

    def __call__(self, labels):
        labels = self._binarise_labels(labels)
        return labels


# for 3d volumes like SegA
class LabelResizeTrafoFor3dInputs(LabelTrafoToBinary):
    def __init__(self, desired_shape, padding="constant", switch_last_axes=False):
        self.desired_shape = desired_shape
        self.padding = padding
        self.switch_last_axes = switch_last_axes

    def __call__(self, labels):
        # binarize the samples
        labels = self._binarise_labels(labels)

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

        if self.switch_last_axes:
            labels = labels.transpose(0, 2, 1)

        return labels
