import argparse

import torch

import torch_em

from micro_sam.models import peft_sam
from micro_sam.training.semantic_sam_trainer import CustomDiceLoss

from medico_sam.loss import CustomCombinedLoss
from medico_sam.util import get_semantic_sam_model


def finetune_semantic_sam(args):
    """Code for finetuning SAM on medical datasets for semantic segmentation."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    from common import get_num_classes, DATASETS_2D, DATASETS_3D, get_dataloaders

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    dataset = args.dataset
    model_type = args.model_type
    checkpoint_path = args.checkpoint  # override this to start training from a custom checkpoint
    num_classes = get_num_classes(dataset)  # 1 background class and 'n' semantic foreground classes

    # TODO: Refactor the code below in future to `get_semantic_sam_model` function.
    if dataset in DATASETS_2D:  # training 2d semantic segmentation models with additional segmentation decoder.
        # the patch shape for 2d semantic segmentation training
        patch_shape = (1024, 1024)

        # Get 2d semantic segmentation model based on UNETR.
        model = get_semantic_sam_model(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            ndim=2,
            device=device,
            peft_kwargs=None if args.lora_rank is None else {
                "peft_module": peft_sam.LoRASurgery,  # the chosen PEFT method (LoRA) for finetuning
                "rank": args.lora_rank,  # whether to use LoRA for finetuning and the rank used for LoRA
            },
        )

    elif dataset in DATASETS_3D:
        # the patch shape for 3d semantic segmentation training
        patch_shape = (16, 512, 512)

        # Get 3d semantic segmentation model based on UNETR.
        model = get_semantic_sam_model(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            ndim=3,
            peft_kwargs={"rank": args.lora_rank},
            device=device,
        )

    else:
        raise ValueError(f"'{dataset}' is not a valid dataset name or not part of our experiments yet.")

    model.to(device)
    checkpoint_name = f"{model_type}/{dataset}_semanticsam"

    # all the stuff we need for training
    learning_rate = 1e-4
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10}

    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path, dataset_name=dataset)

    # And the trainer for semantic segmentation and validation (based on `torch-em`).
    trainer = torch_em.default_segmentation_trainer(
        name=checkpoint_name,
        save_root=args.save_root,
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
    trainer.fit(iterations=int(args.iterations), overwrite_training=False)


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the semantic segmentation tasks.")
    parser.add_argument(
        "-d", "--dataset", required=True, help="The name of medical dataset for semantic segmentation."
    )
    parser.add_argument(
        "-i", "--input_path", default="/mnt/vast-nhr/projects/cidas/cca/data",
        help="The filepath to the medical data. If the data does not exist yet it will be downloaded."
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
        "--iterations", type=int, default=int(5e4), help="For how many iterations should the model be trained?"
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, help="The pretrained weights to initialize the model."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=None,
        help="Whether to use LoRA with provided rank for finetuning SAM for semantic segmentation."
    )
    parser.add_argument(
        "--dice_weight", type=float, default=0.5, help="The weight for dice loss with combined cross entropy loss."
    )
    args = parser.parse_args()
    finetune_semantic_sam(args)


if __name__ == "__main__":
    main()
