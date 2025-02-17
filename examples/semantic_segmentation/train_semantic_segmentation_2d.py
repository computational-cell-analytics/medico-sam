import os
from typing import Union, Tuple, Literal

import torch

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_oimhs_loader

import micro_sam.training as sam_training
from micro_sam.training.util import ConvertToSemanticSamInputs

from medico_sam.util import LinearWarmUpScheduler


# DATA_ROOT = "data"
DATA_ROOT = "/media/anwai/ANWAI/data"


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

    # training settings:
    model_type = "vit_b"  # override this to your desired choice of Segment Anything model.
    checkpoint_path = None  # override this to start training from a custom checkpoint
    num_classes = 5  # 1 background class and 'n' semantic foreground classes
    device = "cuda" if torch.cuda.is_available() else "cpu"  # device to train the model on.
    checkpoint_name = "oimhs_semantic_sam"  # the name for storing the checkpoints.
    patch_shape = (1024, 1024)  # the patch shape for 2d semantic segmentation training

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        flexible_load_checkpoint=True,
        num_multimask_outputs=num_classes,
    )
    model.to(device)

    # all the stuff we need for training
    n_epochs = 100
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    mscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5)
    scheduler = LinearWarmUpScheduler(optimizer, warmup_epochs=4, main_scheduler=mscheduler)

    # Get the dataloaders
    train_loader = get_data_loaders(os.path.join(DATA_ROOT, "oimhs"), "train", patch_shape)
    val_loader = get_data_loaders(os.path.join(DATA_ROOT, "oimhs"), "val", patch_shape)

    # this class creates all the training data for a batch (inputs and labels)
    convert_inputs = ConvertToSemanticSamInputs()

    # the trainer which performs the semantic segmentation training and validation (implemented using "torch_em")
    trainer = sam_training.SemanticSamTrainer(
        name=checkpoint_name,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        log_image_interval=100,
        mixed_precision=True,
        compile_model=False,
        convert_inputs=convert_inputs,
        num_classes=num_classes,
        dice_weight=0.5,
    )
    trainer.fit(epochs=n_epochs)


def main():
    finetune_semantic_sam_2d()


if __name__ == "__main__":
    main()
