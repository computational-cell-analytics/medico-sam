import os
from typing import Union, Tuple, Literal

import torch

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_oasis_loader

import micro_sam.training as sam_training
from micro_sam.models import sam_3d_wrapper
from micro_sam.training.util import ConvertToSemanticSamInputs

from medico_sam.util import LinearWarmUpScheduler
from medico_sam.transform import RawTrafoFor3dInputs


DATA_ROOT = "data"


def get_data_loaders(data_path: Union[os.PathLike, str], split: Literal["train", "val"], patch_shape: Tuple[int, int]):
    """Return train or val data loader for finetuning SAM for 3d semantic segmentation.

    The data loader must be a torch data loader that returns `x, y` tensors,
    where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask semantic segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reserved for backgrund, and the other IDs must map to different classes.

    NOTE: The spatial shapes of images and corresponding labels are expected to be:
    `images: (B, 3, C, Y, X)`, `labels: (B, 1, C, Y, X)` to train the 3d semantic segmentation model.

    Here, we use `torch_em` based data loader, for creating a suitable data loader from
    OIMHS data. You can either see `torch_em.data.datasets.medical.get_oimhs_loader` for adapting
    this on your own data or write a suitable torch dataloader yourself.
    """
    # Get the dataloader.
    loader = get_oasis_loader(
        path=data_path,
        batch_size=1,
        patch_shape=patch_shape,
        split=split,
        resize_inputs=True,
        download=True,
        sampler=MinInstanceSampler(),
        raw_transform=RawTrafoFor3dInputs(),
        pin_memory=True,
        shuffle=True,
    )

    return loader


def finetune_semantic_sam_3d():
    """Scripts for training a 3d semantic segmentation model on medical datasets."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"  # device to train the model on.

    # training settings:
    model_type = "vit_b"  # override this to your desired choice of Segment Anything model.
    checkpoint_path = None  # override this to start training from a custom checkpoint
    num_classes = 5  # 1 background class and 'n' semantic foreground classes
    checkpoint_name = "_semantic_sam"  # the name for storing the checkpoints.
    patch_shape = (32, 512, 512)  # the patch shape for 3d semantic segmentation training

    # get the trainable segment anything model
    model = sam_3d_wrapper.get_sam_3d_model(
        model_type=model_type,
        device=device,
        n_classes=num_classes,
        image_size=patch_shape[-1],
        checkpoint_path=checkpoint_path,
    )

    # all the stuff we need for training
    n_epochs = 100
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    mscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5)
    scheduler = LinearWarmUpScheduler(optimizer, warmup_epochs=4, main_scheduler=mscheduler)

    # Get the dataloaders
    train_loader = get_data_loaders(os.path.join(DATA_ROOT, "oasis"), "train", patch_shape)
    val_loader = get_data_loaders(os.path.join(DATA_ROOT, "oasis"), "val", patch_shape)

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
        mixed_precision=True,
        compile_model=False,
        convert_inputs=convert_inputs,
        num_classes=num_classes,
        dice_weight=0.5,
    )
    trainer.fit(epochs=n_epochs)


def main():
    finetune_semantic_sam_3d()


if __name__ == "__main__":
    main()
