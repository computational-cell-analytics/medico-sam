import argparse

import torch

import torch_em

from micro_sam.models import peft_sam, sam_3d_wrapper
from micro_sam.instance_segmentation import get_unetr
from micro_sam.util import get_sam_model, _load_checkpoint

from medico_sam.models.unetr3d import SimpleUNETR3D


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

    if dataset in DATASETS_2D:  # training 2d semantic segmentation models with additional segmentation decoder.
        # Get the basic 2d UNETR model.
        predictor, state = get_sam_model(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            return_state=True,
            peft_kwargs=None if args.lora_rank is None else {
                "peft_module": peft_sam.LoRASurgery,  # the chosen PEFT method (LoRA) for finetuning
                "rank": args.lora_rank,  # whether to use LoRA for finetuning and the rank used for LoRA
            },
        )
        decoder_state = state.get("decoder_state", None)

        # We remove `out_conv`-related parameters and let it initialize from scratch.
        if decoder_state:
            for k in list(state["decoder_state"].keys()):
                if k.startswith("out_conv"):
                    del decoder_state[k]

        # the patch shape for 2d semantic segmentation training
        patch_shape = (1024, 1024)

        # Finally, get the 2d UNETR model.
        model = get_unetr(
            image_encoder=predictor.model.image_encoder,
            decoder_state=decoder_state,
            out_channels=num_classes,
            flexible_load_checkpoint=True,
        )

    elif dataset in DATASETS_3D:
        # Get the 3d wrapped SAM image encoder.
        sam_3d = sam_3d_wrapper.get_sam_3d_model(
            device=device,
            n_classes=num_classes,
            image_size=512,
            lora_rank=args.lora_rank,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
        )
        state, _ = _load_checkpoint(checkpoint_path=checkpoint_path)
        decoder_state = state.get("decoder_state", None)

        # We remove `out_conv`-related parameters and let it initialize from scratch.
        if decoder_state:
            for k in list(state["decoder_state"].keys()):
                if k.startswith("out_conv"):
                    del decoder_state[k]

        # TODO: Puzzle in the 2d decoder weights!

        # the patch shape for 3d semantic segmentation training
        patch_shape = (16, 512, 512)

        # Finally, get the 3d UNETR model.
        print(num_classes)
        model = SimpleUNETR3D(
            encoder=sam_3d.sam_model.image_encoder,
            out_channels=num_classes,
            final_activation="Sigmoid",
        )

    else:
        raise ValueError(f"'{dataset}' is not a valid dataset name or not part of our experiments yet.")

    model.to(device)
    checkpoint_name = f"{model_type}/{dataset}_semanticsam"

    # all the stuff we need for training
    learning_rate = 1e-4
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 5}

    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path, dataset_name=dataset)

    # And the trainer for semantic segmentation and validation (based on `torch-em`).
    trainer = torch_em.default_segmentation_trainer(
        name=checkpoint_name,
        save_root=args.save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device="cuda",
        learning_rate=learning_rate,
        scheduler_kwargs=scheduler_kwargs,
        log_image_interval=100,
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
        "--iterations", type=int, default=int(1e5), help="For how many iterations should the model be trained?"
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
