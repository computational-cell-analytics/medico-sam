import os

import torch

import torch_em
from torch_em.data.datasets.medical.psfhs import get_psfhs_paths

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model


DATA_FOLDER = "data"


def get_dataloader(split, patch_shape, batch_size):
    """Return train or val data loader for finetuning SAM.

    The data loader must be a torch data loader that returns `x, y` tensors,
    where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive

    Here, we use `torch_em.default_segmentation_loader` for creating a suitable data loader from
    the example hela data. You can either adapt this for your own data (see comments below)
    or write a suitable torch dataloader yourself.
    """
    assert split in ("train", "val")
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # This will download the image and segmentation data for training.
    image_paths, gt_paths = get_psfhs_paths(path=DATA_FOLDER, split="train", download=True)

    # torch_em.default_segmentation_loader is a convenience function to build a torch dataloader
    # from image data and labels for training segmentation models.
    # It supports image data in various formats. Here, we load image data and labels from the two
    # list of filepaths for images and corresponding labels with mha images that were downloaded by
    # the example data functionality, by specifying `raw_key` and `label_key` as 'None'.
    # This means all images in the respective folders that end with .tif will be loaded.
    # The function supports many other file formats. For example, if you have tifs or tif stacks with multiple slices
    # instead of multiple tif images in a foldder, then you can pass raw_key=label_key=None.

    # Load individual images and corresponding labels.
    raw_key, label_key = None, None

    # Here, we select the first 100 images for the train split and the next 50 images for the val split.
    if split == "train":
        image_paths, gt_paths = image_paths[:100], gt_paths[:100]
    else:
        image_paths, gt_paths = image_paths[100:150], gt_paths[100:150]

    label_transform = torch_em.transform.label.connected_components

    loader = torch_em.default_segmentation_loader(
        raw_paths=image_paths,
        raw_key=raw_key,
        label_paths=gt_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        batch_size=batch_size,
        ndim=2,
        is_seg_dataset=False,
        label_transform=label_transform,
        num_workers=8,
        shuffle=True,
        raw_transform=sam_training.identity,
    )
    return loader


def run_training(checkpoint_name, model_type):
    """Run the actual model training."""

    # All hyperparameters for training.
    batch_size = 1  # the training batch size
    patch_shape = (1, 512, 512)  # the size of patches for training
    n_objects_per_batch = 25  # the number of objects per batch that will be sampled
    device = torch.device("cuda")  # the device/GPU used for training

    # Get the dataloaders.
    train_loader = get_dataloader("train", patch_shape, batch_size)
    val_loader = get_dataloader("val", patch_shape, batch_size)

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=False,
        device=device,
    )


def export_model(checkpoint_name, model_type):
    """Export the trained model."""
    # export the model after training so that it can be used by the rest of the micro_sam library
    export_path = "./finetuned_psfhs_model.pth"
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path, model_type=model_type, save_path=export_path,
    )


def main():
    """Finetune a Segment Anything model.

    This example uses image data and segmentations from the PSFHS challenge,
    but can easily be adapted for other data (including data you have annoated with micro_sam beforehand).
    """
    # The model_type determines which base model is used to initialize the weights that are finetuned.
    # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
    model_type = "vit_b"

    # The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
    checkpoint_name = "sam_psfhs"

    run_training(checkpoint_name, model_type)
    export_model(checkpoint_name, model_type)


if __name__ == "__main__":
    main()
