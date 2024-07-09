from torch.optim.lr_scheduler import _LRScheduler


def get_medico_sam_model(
    model_type, device=None, checkpoint_path=None, use_sam_med2d=False, use_sam3d=False, encoder_adapter=False, **kwargs
):
    model_kwargs = {
        "model_type": model_type,
        "device": device,
        "checkpoint_path": checkpoint_path,
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

        for k, v in kwargs.items():
            model_kwargs[k] = v

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
