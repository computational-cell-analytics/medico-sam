import os
from typing import Union, Tuple

import torch

import micro_sam.training as sam_training
from micro_sam.models import sam_3d_wrapper
from micro_sam.training.util import ConvertToSemanticSamInputs

from medico_sam.util import LinearWarmUpScheduler


DATA_ROOT = "data"


def get_data_loaders(data_path: Union[os.PathLike, str], patch_shape: Tuple[int, int]):
    """
    """
    return train_loader, val_loader


def finetune_semantic_sam_3d(num_classes: int):
    """Scripts for training a 3d semantic segmentation model on medical datasets."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu

    # training settings:
    model_type = "vit_b"  # override this to your desired choice of Segment Anything model.
    checkpoint_path = None  # override this to start training from a custom checkpoint
    num_classes = ...  # 1 background class and 'n' semantic foreground classes
    device = "cuda" if torch.cuda.is_available() else "cpu"  # device to train the model on.
    checkpoint_name = "_semantic_sam"  # the name for storing the checkpoints.
    patch_shape = (32, 512, 512)  # the patch shape for 3d semantic segmentation training

    # get the trainable segment anything model
    model = sam_3d_wrapper.get_sam_3d_model(
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
    train_loader, val_loader = get_data_loaders(data_path=DATA_ROOT, patch_shape=patch_shape)

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
