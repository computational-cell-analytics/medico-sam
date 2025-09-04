import os
from typing import Union, Tuple, Literal

import torch

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_oimhs_loader

import micro_sam.training as sam_training
from micro_sam.training.semantic_sam_trainer import CustomDiceLoss

from medico_sam.loss import CustomCombinedLoss
from medico_sam.util import get_semantic_sam_model


DATA_ROOT = "data"


def get_data_loaders(data_path: Union[os.PathLike, str], split: Literal["train", "val"], patch_shape: Tuple[int, int]):
    """Return train or val data loader for finetuning SAM for 2d semantic segmentation.

    The data loader must be a torch data loader that returns `x, y` tensors,
    where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask semantic segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reserved for backgrund, and the other IDs must map to different classes.

    NOTE: The spatial shapes of images and corresponding labels are expected to be:
    `images: (B, C, Y, X)`, `labels: (B, 1, Y, X)` to train the 2d semantic segmentation model,
    where C corresponds to the number of channels in input data.
    (eg. the OCT images used in the example below are images with three channels)

    Here, we use `torch_em` based data loader, for creating a suitable data loader from
    OIMHS data. You can either see `torch_em.data.datasets.medical.get_oimhs_loader` for adapting
    this on your own data or write a suitable torch dataloader yourself.
    """
    # Get the dataloader.
    loader = get_oimhs_loader(
        path=data_path,
        batch_size=1,
        patch_shape=patch_shape,
        split=split,
        resize_inputs=True,
        download=True,
        sampler=MinInstanceSampler(),
        raw_transform=sam_training.identity,
        pin_memory=True,
        shuffle=True,
    )

    return loader


def finetune_semantic_sam_2d():
    """Scripts for training a 2d semantic segmentation model on medical datasets."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"  # device to train the model on.

    # training settings:
    model_type = "vit_b_medical_imaging"  # override this to your desired choice of Segment Anything model.
    checkpoint_path = None  # override this to start training from a custom checkpoint
    num_classes = 5  # 1 background class and 'n' semantic foreground classes
    checkpoint_name = "oimhs_semantic_sam"  # the name for storing the checkpoints.
    patch_shape = (1024, 1024)  # the patch shape for 2d semantic segmentation training

    # Get the UNETR-style model with pretrained image encoder and segmentation decoder.
    model = get_semantic_sam_model(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        ndim=2,
        device=device,
        init_decoder_weights=True,
    )
    model.to(device)

    # all the stuff we need for training
    n_epochs = 100
    learning_rate = 1e-4
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10}

    # Get the dataloaders
    train_loader = get_data_loaders(os.path.join(DATA_ROOT, "oimhs"), "train", patch_shape)
    val_loader = get_data_loaders(os.path.join(DATA_ROOT, "oimhs"), "val", patch_shape)

    # And the trainer for semantic segmentation training and validation (based on `torch-em`).
    trainer = torch_em.default_segmentation_trainer(
        name=checkpoint_name,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        learning_rate=learning_rate,
        scheduler_kwargs=scheduler_kwargs,
        log_image_interval=10,
        mixed_precision=True,
        compile_model=False,
        loss=CustomCombinedLoss(num_classes=num_classes),
        metric=CustomDiceLoss(num_classes=num_classes),
    )
    trainer.fit(epochs=n_epochs)


def main():
    finetune_semantic_sam_2d()


if __name__ == "__main__":
    main()
