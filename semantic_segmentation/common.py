import numpy as np
from math import ceil, floor

from torch.optim.lr_scheduler import _LRScheduler

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


# learning rate scheduler using warmup
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
            return self.main_scheduler.optimizer.param_groups[0]['lr']

    def step(self, metrics=None, epoch=None):
        if not self.is_warmup_finished:
            super().step()
        else:
            self.main_scheduler.step(metrics, epoch)

    def _get_closed_form_lr(self):
        return self.get_lr()
