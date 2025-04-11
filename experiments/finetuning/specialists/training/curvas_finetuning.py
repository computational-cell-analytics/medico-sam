import argparse

import torch

from torch_em.loss import DiceLoss
from torch_em.data import MinTwoInstanceSampler
from torch_em.data.datasets import get_curvas_loader

from micro_sam.instance_segmentation import get_unetr
from micro_sam.training import joint_sam_trainer as joint_trainers
from micro_sam.training.util import get_trainable_sam_model, ConvertToSamInputs

from medico_sam.transform import LabelTransformJointTraining, RawTrafnsformJointTraining


def get_dataloaders(patch_shape, data_path):
    """This returhs the CURVAS data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/medical/curvas.py
    It will automatically download the CURVAS data.

    NOTE: To replace this with another data loader, you need to return a torch data loader
    that returns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID in the first channel,
    and the second channel with all object masks as foreground for binary semantic segmentation.
    Important: the ID 0 is reserved for background, and the IDs must be consecutive.
    """
    label_transform = LabelTransformJointTraining()
    raw_transform = RawTrafnsformJointTraining(modality="CT")
    sampler = MinTwoInstanceSampler()

    train_loader = get_curvas_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=4,
        split="train",
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=sampler,
        raw_transform=raw_transform,
        label_transform=label_transform,
        num_workers=16,
        n_samples=200,
    )
    val_loader = get_curvas_loader(
        path=data_path,
        patch_shape=patch_shape,
        batch_size=1,
        split="val",
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=sampler,
        raw_transform=raw_transform,
        label_transform=label_transform,
        num_workers=16,
        n_samples=100,
    )

    return train_loader, val_loader


def finetune_curvas(args):
    """Example code for finetuning SAM on CURVAS."""

    # override this (below) if you have some more complex set-up and need to specify the exact gpu.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    lr = 1e-5  # the learning rate.
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (1, 512, 512)  # the patch shape for training
    n_objects_per_batch = args.n_objects  # the number of objects per batch that will be sampled (default: 25)
    freeze_parts = args.freeze  # override this to freeze different parts of the model
    checkpoint_name = f"{args.model_type}/curvas_sam"

    # Get the trainable segment anything model.
    model, state = get_trainable_sam_model(
        model_type=model_type,
        device=device,
        freeze=freeze_parts,
        checkpoint_path=checkpoint_path,
        return_state=True,
    )

    # This class creates all the training data for a batch (inputs, prompts and labels).
    convert_inputs = ConvertToSamInputs(transform=model.transform, box_distortion_factor=0.05)

    # Get the UNETR.
    unetr = get_unetr(
        image_encoder=model.sam.image_encoder,
        decoder_state=state.get("decoder_state", None),
        device=device,
        out_channels=1,
    )

    # Get the parameters for SAM and the decoder from UNETR.
    model_params = [params for params in model.parameters()]  # sam parameters
    for param_name, params in unetr.named_parameters():  # unetr's decoder parameters
        if not param_name.startswith("encoder"):
            model_params.append(params)

    # all the stuff we need for training.
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path)
    optimizer = torch.optim.AdamW(model_params, lr=lr)
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10, "verbose": True}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **scheduler_kwargs)

    # The trainer which performs training and validation.
    semantic_seg_loss = DiceLoss()
    trainer = joint_trainers.JointSamTrainer(
        name=checkpoint_name,
        save_root=args.save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        logger=joint_trainers.JointSamLogger,
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        n_sub_iteration=8,
        compile_model=False,
        unetr=unetr,
        instance_loss=semantic_seg_loss,
        instance_metric=semantic_seg_loss,
        early_stopping=None,
        mask_prob=0.5,
    )
    trainer.fit(iterations=args.iterations)


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
