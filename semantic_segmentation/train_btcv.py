import os
import argparse
from glob import glob
from natsort import natsorted

import torch

import torch_em
from torch_em.data import MinInstanceSampler

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model
from micro_sam.sam_3d_wrapper import get_3d_sam_model
from micro_sam.training.util import ConvertToSemanticSamInputs

from common import RawTrafoFor3dInputs


def get_dataloaders(patch_shape, data_path):
    """This returns the btcv data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/medical/btcv.py
    It will not automatically download the BTCV data. Take a look at `get_btcv_dataset`.

    NOTE: The step below is done to obtain the BTCV dataset in splits.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    kwargs = {}
    kwargs["raw_transform"] = RawTrafoFor3dInputs()
    kwargs["sampler"] = MinInstanceSampler(min_num_instances=8)

    train_image_paths = natsorted(glob(os.path.join(data_path, "imagesTr", "*_train_0000.nii.gz")))
    train_gt_paths = natsorted(glob(os.path.join(data_path, "labelsTr", "*_train.nii.gz")))
    val_image_paths = natsorted(glob(os.path.join(data_path, "imagesTr", "*_val_0000.nii.gz")))
    val_gt_paths = natsorted(glob(os.path.join(data_path, "labelsTr", "*_val.nii.gz")))

    train_dataset = torch_em.default_segmentation_dataset(
        raw_paths=train_image_paths,
        raw_key="data",
        label_paths=train_gt_paths,
        label_key="data",
        is_seg_dataset=True,
        ndim=3,
        patch_shape=patch_shape,
        **kwargs
    )
    val_dataset = torch_em.default_segmentation_dataset(
        raw_paths=val_image_paths,
        raw_key="data",
        label_paths=val_gt_paths,
        label_key="data",
        is_seg_dataset=True,
        ndim=3,
        patch_shape=patch_shape,
        **kwargs
    )

    batch_size = 1
    train_loader = torch_em.get_data_loader(
        dataset=train_dataset, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True,
    )
    val_loader = torch_em.get_data_loader(
        dataset=val_dataset, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True,
    )

    return train_loader, val_loader


def finetune_btcv(args):
    """Code for finetuning SAM on BTCV for semantic segmentation."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = args.checkpoint  # override this to start training from a custom checkpoint
    patch_shape = (32, 512, 512)  # the patch shape for training
    num_classes = 14  # 1 background class and 13 semantic foreground classes

    lora_rank = 4 if args.use_lora else None
    freeze_encoder = True if lora_rank is None else False

    # get the trainable segment anything model
    model = get_3d_sam_model(
        device=device,
        n_classes=num_classes,
        image_size=512,
        checkpoint_path=checkpoint_path,
        freeze_encoder=freeze_encoder,
        lora_rank=lora_rank,
    )
    model.to(device)

    # all the stuff we need for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=3, verbose=True)
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = ConvertToSemanticSamInputs()

    lora_str = "frozen" if lora_rank is None else f"lora{lora_rank}"
    checkpoint_name = f"{args.model_type}_3d_{lora_str}/btcv_semanticsam"

    # the trainer which performs the semantic segmentation training and validation (implemented using "torch_em")
    trainer = sam_training.semantic_sam_trainer.SemanticSamTrainer3D(
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
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the BTCV dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/share/cidas/cca/nnUNetv2/nnUNet_raw/Dataset301_BTCV/",
        help="The filepath to the BTCV data. If the data does not exist yet it will be downloaded."
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
        "--export_path", "-e",
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools."
    )
    parser.add_argument(
        "--save_every_kth_epoch", type=int, default=None,
        help="To save every kth epoch while fine-tuning. Expects an integer value."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, help="The pretrained weights to initialize the model."
    )
    parser.add_argument(
        "--use_lora", action="store_true", help="Whether to use LoRA for finetuning SAM for semantic segmentation."
    )
    args = parser.parse_args()
    finetune_btcv(args)


if __name__ == "__main__":
    main()
