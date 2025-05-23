import os
import argparse

import torch

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_sa_med2d_loader

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model


def get_dataloaders(data_path, patch_shape):
    raw_transform = sam_training.identity

    train_loader = get_sa_med2d_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=8,
        split="train",
        resize_inputs=True,
        num_workers=64,
        shuffle=True,
        pin_memory=True,
        n_fraction_per_dataset=0.1,
        raw_transform=raw_transform,
        sampler=MinInstanceSampler(),
    )
    val_loader = get_sa_med2d_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=8,
        split="val",
        resize_inputs=True,
        num_workers=64,
        pin_memory=True,
        n_fraction_per_dataset=0.1,
        raw_transform=raw_transform,
        sampler=MinInstanceSampler(),
    )
    return train_loader, val_loader


def finetune_medical_generalist(args):
    """Code for finetuning SAM in Mazurowski et al.-style on SA-Med2D-20M dataset,
    composed of multiple medical datasets
    """
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (1024, 1024)  # the patch shape for training
    n_objects_per_batch = args.n_objects  # this is the number of objects per batch that will be sampled (default: 25)
    freeze_parts = None  # override this to freeze one or more of these backbones

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts
    )
    model.to(device)

    # all the stuff we need for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9, verbose=True)
    train_loader, val_loader = get_dataloaders(data_path=args.input_path, patch_shape=patch_shape)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = sam_training.ConvertToSamInputs(transform=model.transform, box_distortion_factor=0.025)

    checkpoint_name = f"{args.model_type}/medical_generalist_simplesam_single_gpu"

    # the trainer which performs the joint training and validation (implemented using "torch_em")
    trainer = sam_training.SimpleSamTrainer(
        name=checkpoint_name,
        save_root=args.save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        logger=sam_training.SamLogger,
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        compile_model=False,
        # while choosing both the prompting choice, it randomly selects between either of the two per iteration.
        use_points=True,
        use_box=True,
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
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the Medical datasets.")
    parser.add_argument(
        "--input_path", "-i", type=str, default="/scratch/share/cidas/cca/data/sa-med2d",
        help="The filepath to all the respective medical datasets. If the data does not exist yet it will be downloaded"
    )
    parser.add_argument(
        "--model_type", "-m", type=str, default="vit_b",
        help="The model type to use for fine-tuning. Either vit_t, vit_b, vit_l or vit_h."
    )
    parser.add_argument(
        "--save_root", "-s", type=str, default=None,
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run from."
    )
    parser.add_argument(
        "--iterations", type=int, default=int(3e5),
        help="For how many iterations should the model be trained? By default 300k."
    )
    parser.add_argument(
        "--export_path", "-e", type=str, default=None,
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools."
    )
    parser.add_argument(
        "--save_every_kth_epoch", type=int, default=None,
        help="To save every kth epoch while fine-tuning. Expects an integer value."
    )
    parser.add_argument(
        "--n_objects", type=int, default=5, help="The number of instances (objects) per batch used for finetuning."
    )
    args = parser.parse_args()
    finetune_medical_generalist(args)


if __name__ == "__main__":
    main()
