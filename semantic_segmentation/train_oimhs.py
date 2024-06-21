import os
import argparse

import torch

from torch_em.data import MinInstanceSampler
from torch_em.transform.label import OneHotTransform
from torch_em.data.datasets.medical import get_oimhs_loader

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model
from micro_sam.training.util import ConvertToSemanticSamInputs


def get_dataloaders(patch_shape, data_path):
    """This returns the camus data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/medical/oimhs.py
    It will not automatically download the OIMHS data. Take a look at `get_oimhs_dataset`.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    raw_transform = sam_training.identity
    sampler = MinInstanceSampler(min_num_instances=5)
    label_transform = OneHotTransform(class_ids=[0, 1, 2, 3, 4])

    train_loader = get_oimhs_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=8,
        split="train",
        resize_inputs=True,
        raw_transform=raw_transform,
        num_workers=16,
        shuffle=True,
        sampler=sampler,
        pin_memory=True,
        label_transform=label_transform,
    )
    val_loader = get_oimhs_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=1,
        split="val",
        resize_inputs=True,
        raw_transform=raw_transform,
        num_workers=16,
        sampler=sampler,
        pin_memory=True,
        label_transform=label_transform,
    )
    return train_loader, val_loader


def finetune_oimhs(args):
    """Code for finetuning SAM on OIMHS in "micro_sam"-based MedSAM reimplementation"""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (1024, 1024)  # the patch shape for training
    freeze_parts = args.freeze  # override this to freeze different parts of the model
    num_classes = 5  # 1 background class and 4 semantic foreground classes

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,
        flexible_load_checkpoint=True,
        num_multimask_outputs=num_classes,
    )
    model.to(device)

    # all the stuff we need for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, verbose=True)
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path)

    train_loader.dataset.max_sampling_attempts = 10000
    val_loader.dataset.max_sampling_attempts = 10000

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = ConvertToSemanticSamInputs()

    checkpoint_name = f"{args.model_type}/oimhs_semanticsam"

    # the trainer which performs the semantic segmentation training and validation (implemented using "torch_em")
    trainer = sam_training.SemanticSamTrainer(
        name=checkpoint_name,
        save_root=args.save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        log_image_interval=10,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        num_classes=num_classes,
        compile_model=False,
    )
    trainer.fit(args.iterations, save_every_kth_epoch=args.save_every_kth_epoch)
    if args.export_path is not None:
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            save_path=args.export_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the OIMHS dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/share/cidas/cca/data/oimhs/",
        help="The filepath to the OIMHS data. If the data does not exist yet it will be downloaded."
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
        "--iterations", type=int, default=int(1e4),
        help="For how many iterations should the model be trained?"
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
        "--save_every_kth_epoch", type=int, default=None,
        help="To save every kth epoch while fine-tuning. Expects an integer value."
    )
    args = parser.parse_args()
    finetune_oimhs(args)


if __name__ == "__main__":
    main()
