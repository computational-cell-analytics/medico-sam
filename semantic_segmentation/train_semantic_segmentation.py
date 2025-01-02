import argparse

import torch

import micro_sam.training as sam_training
from micro_sam.models import peft_sam, sam_3d_wrapper
from micro_sam.training.util import ConvertToSemanticSamInputs

from medico_sam.util import LinearWarmUpScheduler


def finetune_semantic_sam(args):
    """Code for finetuning SAM on medical datasets for semantic segmentation."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    from common import get_num_classes, DATASETS_2D, DATASETS_3D

    if args.uno:
        from common_uno import get_dataloaders
    else:
        from common import get_dataloaders

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    dataset = args.dataset
    model_type = args.model_type
    checkpoint_path = args.checkpoint  # override this to start training from a custom checkpoint
    freeze_parts = args.freeze  # override this to freeze different parts of the model
    num_classes = get_num_classes(dataset)  # 1 background class and 'n' semantic foreground classes

    if dataset in DATASETS_2D:
        patch_shape = (1024, 1024)  # the patch shape for 2d semantic segmentation training
        peft_kwargs = {
            "peft_module": peft_sam.LoRASurgery,  # the chosen PEFT method (LoRA) for finetuning
            "rank": args.lora_rank,  # whether to use LoRA for finetuning and the rank used for LoRA
        }
        # get the trainable segment anything model
        model = sam_training.get_trainable_sam_model(
            model_type=model_type,
            device=device,
            checkpoint_path=checkpoint_path,
            freeze=freeze_parts,
            flexible_load_checkpoint=True,
            num_multimask_outputs=num_classes,
            peft_kwargs=peft_kwargs if args.lora_rank is not None else None,
        )
        model.to(device)
        checkpoint_name = f"{args.model_type}/{dataset}_semanticsam"

    elif dataset in DATASETS_3D:
        patch_shape = (32, 512, 512)  # the patch shape for 3d semantic segmentation training
        model = sam_3d_wrapper.get_sam_3d_model(
            device=device,
            n_classes=num_classes,
            image_size=patch_shape[-1],
            checkpoint_path=checkpoint_path,
            freeze_encoder=args.lora_rank is None and (freeze_parts and "image_encoder" in freeze_parts),
            lora_rank=args.lora_rank,
        )
        model.to(device)
        if args.lora_rank is not None:
            ft_name = f"lora_{args.lora_rank}"
        else:
            ft_name = "frozen" if (freeze_parts and "image_encoder" in freeze_parts) else "all"
        checkpoint_name = f"{model_type}_3d_{ft_name}/{args.dataset}_semanticsam"

    else:
        raise ValueError(f"'{dataset}' is not a valid dataset name.")

    # all the stuff we need for training
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    mscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5, verbose=True)
    scheduler = LinearWarmUpScheduler(optimizer, warmup_epochs=4, main_scheduler=mscheduler)

    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path, dataset_name=dataset)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = ConvertToSemanticSamInputs()

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
        "--iterations", type=int, default=int(1e5), help="For how many iterations should the model be trained?"
    )
    parser.add_argument(
        "--freeze", type=str, nargs="+", default=None, help="Which parts of the model to freeze for finetuning."
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
    parser.add_argument(
        "--uno", action="store_true", help="Whether to train for semantic segmentation on one image only."
    )

    args = parser.parse_args()
    finetune_semantic_sam(args)


if __name__ == "__main__":
    main()
