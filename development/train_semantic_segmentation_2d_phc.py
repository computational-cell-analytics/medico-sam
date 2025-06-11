import os
from glob import glob
from pathlib import Path
from natsort import natsorted

import numpy as np
import imageio.v3 as imageio

import torch

import torch_em
from torch_em.data import MinInstanceSampler

import micro_sam.training as sam_training
from micro_sam.training.util import ConvertToSemanticSamInputs

from medico_sam.util import LinearWarmUpScheduler


# DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"  # Image-label pairs from GitHub simply dropped here.
DATA_ROOT = "/home/anwai/data"

LABEL_MAP = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (255, 165, 0): 3,
}


def get_data_loaders(data_path, split, patch_shape):
    # Get the image and label paths.
    image_paths = natsorted(glob(os.path.join(data_path, "image_*.png")))
    label_paths = natsorted(glob(os.path.join(data_path, "label_*.tif")))

    # Make a simple hot-fix to map labels to single channel before preparing the dataloader.
    if len(label_paths) == 0:
        curr_label_paths = natsorted(glob(os.path.join(data_path, "label_*.png")))  # Get all RGB-style labels.
        for curr_lpath in curr_label_paths:
            label = imageio.imread(curr_lpath)
            flabel = np.zeros(label.shape[:2], dtype=np.uint8)  # Create a single-channel zero array for mapping labels.
            for chan_map, class_id in LABEL_MAP.items():
                flabel[np.all(label == chan_map, axis=-1)] = class_id  # Map known RGB maps consistency to class id.
            imageio.imwrite(str(Path(curr_lpath).with_suffix(".tif")), flabel, compression="zlib")
        label_paths = natsorted(glob(os.path.join(data_path, "label_*.tif")))

    if split == "train":  # Train on first two images
        image_paths, label_paths = image_paths[:2], label_paths[:2]
    else:  # Validate on the last image.
        image_paths, label_paths = image_paths[2:], label_paths[2:]

    # Prepare the dataloader.
    loader = torch_em.default_segmentation_loader(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        batch_size=1,
        is_seg_dataset=False,
        ndim=2,
        with_channels=True,
        patch_shape=patch_shape,
        raw_transform=sam_training.identity,
        sampler=MinInstanceSampler(min_num_instances=4),
    )

    return loader


def finetune_semantic_sam_2d():
    """Scripts for training a 2d semantic segmentation model on a microscopy dataset."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu

    # training settings:
    model_type = "vit_b"  # override this to your desired choice of Segment Anything model.
    checkpoint_path = None  # override this to start training from a custom checkpoint
    num_classes = 4  # 1 background class and 'n' semantic foreground classes
    device = "cuda" if torch.cuda.is_available() else "cpu"  # device to train the model on.
    checkpoint_name = "phc-labelme_semantic_sam"  # the name for storing the checkpoints.
    patch_shape = (512, 512)  # the patch shape for 2d semantic segmentation training

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
    train_loader = get_data_loaders(os.path.join(DATA_ROOT, "phc-labelme"), "train", patch_shape)
    val_loader = get_data_loaders(os.path.join(DATA_ROOT, "phc-labelme"), "val", patch_shape)

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
