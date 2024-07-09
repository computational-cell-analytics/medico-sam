import numpy as np
from math import ceil, floor


class LabelTrafoToBinary:
    def __call__(self, labels):
        labels = (labels > 0).astype(labels.dtype)
        return labels


# for 3d volumes like SegA
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
