import argparse

import torch

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_curvas_loader

import micro_sam.training as sam_training


def get_dataloaders(patch_shape, data_path):
    """
    """
    label_transform = ...  # TODO
    raw_transform = ...  # TODO

    train_loader = get_curvas_loader(
        path=data_path, patch_shape=patch_shape, batch_size=2, split="train", ndim=2, resize_inputs=True,
        download=True, sampler=MinInstanceSampler(min_size=50), raw_transform=None, label_transform=None,
    )
    val_loader = get_curvas_loader(
        path=data_path, patch_shape=patch_shape, batch_size=1, split="val", ndim=2, resize_inputs=True,
        download=True, sampler=MinInstanceSampler(min_size=50), raw_transform=None, label_transform=None,
    )

    return train_loader, val_loader


def finetune_curvas(args):
    """Example code for finetuning SAM on CURVAS."""

    # override this (below) if you have some more complex set-up and need to specify the exact gpu.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (1, 512, 512)  # the patch shape for training
    n_objects_per_batch = args.n_objects  # the number of objects per batch that will be sampled (default: 25)
    freeze_parts = args.freeze  # override this to freeze different parts of the model
    checkpoint_name = f"{args.model_type}/curvas_sam"

    # all the stuff we need for training.
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path)
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10, "verbose": True}

    breakpoint()

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=None,
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,
        device=device,
        lr=1e-5,
        n_iterations=args.iterations,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
        with_segmentation_decoder=True,  # NOTE: this will be utilized for binary semantic segmentation.
    )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the CURVAS dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/mnt/vast-nhr/projects/cidas/cca/data/curvas/",
        help="The filepath to the CURVAS data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_t, vit_b, vit_l or vit_h."
    )
    parser.add_argument(
        "--save_root", "-s",
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run."
    )
    parser.add_argument(
        "--iterations", type=int, default=int(1e5),
        help="For how many iterations should the model be trained? By default 100k."
    )
    parser.add_argument(
        "--export_path", "-e",
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools."
    )
    parser.add_argument(
        "--freeze", type=str, nargs="+", default=None,
        help="Which parts of the model to freeze for finetuning."
    )
    parser.add_argument(
        "--n_objects", type=int, default=25, help="The number of instances (objects) per batch used for finetuning."
    )
    args = parser.parse_args()
    finetune_curvas(args)


if __name__ == "__main__":
    main()
