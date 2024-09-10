import argparse

import torch

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_piccolo_loader

import micro_sam.training as sam_training
from micro_sam.training.util import ConvertToSemanticSamInputs

from medico_sam.util import LinearWarmUpScheduler
from medico_sam.transform.label import LabelTrafoToBinary


def get_dataloaders(patch_shape, data_path):
    """This returns the piccolo data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/medical/piccolo.py
    It will not automatically download the PICCOLO data. Take a look at `get_piccolo_dataset`.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    raw_transform = sam_training.identity
    sampler = MinInstanceSampler()
    label_transform = LabelTrafoToBinary()

    train_loader = get_piccolo_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=8,
        split="train",
        resize_inputs=True,
        raw_transform=raw_transform,
        label_transform=label_transform,
        num_workers=16,
        shuffle=True,
        sampler=sampler,
        pin_memory=True,
    )
    val_loader = get_piccolo_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=1,
        split="validation",
        resize_inputs=True,
        raw_transform=raw_transform,
        label_transform=label_transform,
        num_workers=16,
        sampler=sampler,
        pin_memory=True,
    )
    return train_loader, val_loader


def finetune_piccolo(args):
    """Code for finetuning SAM on PICCOLO for semantic segmentation."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = args.checkpoint  # override this to start training from a custom checkpoint
    patch_shape = (1024, 1024)  # the patch shape for training
    freeze_parts = args.freeze  # override this to freeze different parts of the model
    num_classes = 2  # 1 background class and 1 semantic foreground classes
    lora_rank = args.lora_rank  # whether to use LoRA for finetuning and the rank used for LoRA

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,
        flexible_load_checkpoint=True,
        num_multimask_outputs=num_classes,
        lora_rank=lora_rank,
    )
    model.to(device)

    # all the stuff we need for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    mscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5, verbose=True)
    scheduler = LinearWarmUpScheduler(optimizer, warmup_epochs=4, main_scheduler=mscheduler)
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = ConvertToSemanticSamInputs()

    checkpoint_name = f"{args.model_type}/piccolo_semanticsam"

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
        log_image_interval=50,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        num_classes=num_classes,
        compile_model=False,
        dice_weight=0.8,
    )
    trainer.fit(args.iterations, save_every_kth_epoch=args.save_every_kth_epoch)


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the PICCOLO dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/share/cidas/cca/data/piccolo/",
        help="The filepath to the PICCOLO data. If the data does not exist yet it will be downloaded."
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
        help="For how many iterations should the model be trained?"
    )
    parser.add_argument(
        "--freeze", type=str, nargs="+", default=None,
        help="Which parts of the model to freeze for finetuning."
    )
    parser.add_argument(
        "--save_every_kth_epoch", type=int, default=None,
        help="To save every kth epoch while fine-tuning. Expects an integer value."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, help="The pretrained weights to initialize the model."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=None,
        help="Whether to use LoRA with provided rank for finetuning SAM for semantic segmentation."
    )
    args = parser.parse_args()
    finetune_piccolo(args)


if __name__ == "__main__":
    main()
