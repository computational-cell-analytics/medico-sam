import argparse

import torch

import micro_sam.training as sam_training
from micro_sam.models.sam_3d_wrapper import get_sam_3d_model
from micro_sam.training.util import ConvertToSemanticSamInputs

from common import get_dataloaders, get_num_classes


def finetune_duke_liver(args):
    """Code for finetuning SAM on medical datasets for 3d semantic segmentation."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = args.checkpoint  # override this to start training from a custom checkpoint
    patch_shape = (32, 512, 512)  # the patch shape for training
    freeze_parts = args.freeze  # override this to freeze different parts of the model
    num_classes = get_num_classes(args.dataset)  # 1 background class and 1 semantic foreground class

    # get the trainable segment anything model
    model = get_sam_3d_model(
        device=device,
        n_classes=num_classes,
        image_size=512,
        checkpoint_path=checkpoint_path,
        freeze_encoder=args.lora_rank is None and freeze_parts == "image_encoder",
        lora_rank=args.lora_rank,
    )
    model.to(device)

    # all the stuff we need for training
    learning_rate = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    mscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=3, verbose=True)

    if args.lr_scheduler:
        from medico_sam.util import LinearWarmUpScheduler
        scheduler = LinearWarmUpScheduler(optimizer, warmup_epochs=4, main_scheduler=mscheduler)
    else:
        scheduler = mscheduler

    train_loader, val_loader = get_dataloaders(
        patch_shape=patch_shape, data_path=args.input_path, dataset_name=args.dataset,
    )

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = ConvertToSemanticSamInputs()

    ft_name = "frozen" if args.lora_rank is None else f"lora_{args.lora_rank}"
    checkpoint_name = f"{model_type}_3d_{ft_name}/{args.dataset}_semanticsam"

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
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        num_classes=num_classes,
        compile_model=False,
        dice_weight=args.dice_weight,
    )
    trainer.fit(args.iterations)


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the Duke Liver dataset.")
    parser.add_argument(
        "-d", "--dataset", required=True, help="The name of medical dataset for semantic segmentation."
    )
    parser.add_argument(
        "--input_path", "-i", default="/scratch/share/cidas/cca/data",
        help="The filepath to the Duke Liver data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_t, vit_b, vit_l or vit_h."
    )
    parser.add_argument(
        "--save_root", "-s", default=None,
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
        "-c", "--checkpoint", type=str, default=None, help="The pretrained weights to initialize the model."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=None,
        help="Whether to use LoRA with provided rank for finetuning SAM for semantic segmentation."
    )
    parser.add_argument(
        "--dice_weight", type=float, default=1, help="The weight for dice loss with combined cross entropy loss."
    )
    parser.add_argument(
        "--lr_scheduler", action="store_true", help="Whether to use linear warmup-based learning rate scheduler."
    )
    args = parser.parse_args()
    finetune_duke_liver(args)


if __name__ == "__main__":
    main()
