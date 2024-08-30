import os
import argparse
from glob import glob

import torch

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.util import update_kwargs_for_resize_trafo
from torch_em import default_segmentation_dataset
from torch_em.transform.label import BoundaryTransform, NoToBackgroundBoundaryTransform

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model
from micro_sam.models.sam_3d_wrapper import get_sam_3d_model
from micro_sam.training.util import ConvertToSemanticSamInputs

from medico_sam.transform.raw import RawResizeTrafoFor3dInputs
from medico_sam.transform.label import LabelResizeTrafoFor3dInputs


def get_mito_dataset(
    paths,
    patch_shape,
    resize_inputs=False,
    download=False,
    raw_key="raw",
    label_key="labels/mitochondria",
    **kwargs,
):
    # image_paths, gt_paths = _get_duke_liver_paths(path=path, split=split, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    #kwargs["label_transform2"] = BoundaryTransform(add_binary_target=True)

    dataset = default_segmentation_dataset(
        raw_paths=paths,
        raw_key=raw_key,
        label_paths=paths,
        label_key=label_key,
        is_seg_dataset=True,
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_mito_loader(dataset: torch.utils.data.Dataset, batch_size, **loader_kwargs) -> torch.utils.data.DataLoader:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **loader_kwargs)
    # monkey patch shuffle attribute to the loader
    loader.shuffle = loader_kwargs.get("shuffle", False)
    return loader


def get_dataloaders(patch_shape, data_path):
    """This returns the duke liver data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/medical/duke_liver.py
    It will not automatically download the Duke Liver data. Take a look at `get_duke_liver_dataset`.

    NOTE: The step below is done to obtain the Duke Liver dataset in splits.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    kwargs = {}
    kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape)
    kwargs["label_transform"] = LabelResizeTrafoFor3dInputs(desired_shape=patch_shape) #BoundaryTransform(add_binary_target=True) 
    kwargs["sampler"] = MinInstanceSampler()
    data_paths = glob(os.path.join(data_path, "**", "*.h5"), recursive=True)
    # filter out all combined - only needed for cristae
    substring = "_combined.h5"
    data_paths = [s for s in data_paths if substring not in s]
    num_data = len(data_paths)
    train_size = int(num_data * .8)
    val_size = int(num_data * .2)  # Optional validation set
    test_size = num_data - train_size - val_size
    remaining = num_data - (train_size + val_size + test_size)
    if remaining > 0:
        val_size += remaining
    data_split = {
        "train": data_paths[:train_size],
        "val": data_paths[train_size:train_size+val_size],
        "test": data_paths[train_size+val_size:]
    }
    train_ds = get_mito_dataset(
        paths=data_split["train"],
        patch_shape=patch_shape,
        **kwargs
    )
    val_ds = get_mito_dataset(
        paths=data_split["val"],
        patch_shape=patch_shape,
        **kwargs
    )
    num_workers = 16
    train_loader = get_mito_loader(
        dataset=train_ds,
        batch_size=1,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = get_mito_loader(
        dataset=val_ds,
        batch_size=1,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    return train_loader, val_loader


def finetune_mito(args):
    """Code for finetuning SAM on Duke Liver for semantic segmentation."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = args.checkpoint  # override this to start training from a custom checkpoint
    patch_shape = (64, 512, 512)  # the patch shape for training
    num_classes = 2  # 1 background class, 1 semantic foreground class

    lora_rank = 4 if args.use_lora else None
    freeze_encoder = True if lora_rank is None else False

    # get the trainable segment anything model
    model = get_sam_3d_model(
        device=device,
        n_classes=num_classes,
        image_size=512,
        checkpoint_path=checkpoint_path,
        freeze_encoder=freeze_encoder,
        lora_rank=lora_rank,
    )
    model.to(device)

    # all the stuff we need for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=3, verbose=True)
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = ConvertToSemanticSamInputs()

    lora_str = "frozen" if lora_rank is None else f"lora{lora_rank}"
    checkpoint_name = f"{args.model_type}_3d_{lora_str}/bs1_ps64"
    print(f"\n{args.save_root}/{checkpoint_name}\n")

    # the trainer which performs the semantic segmentation training and validation (implemented using "torch_em")
    trainer = sam_training.semantic_sam_trainer.SemanticSamTrainer(
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
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the Cryo EM dataset (mitochondria and cristae).")
    parser.add_argument(
        "--input_path", "-i", default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi/",
        help="The filepath to the Duke Liver data. If the data does not exist yet it will be downloaded."
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
    parser.add_argument(
        "--learning_rate", type=float, default=float(5e-4), help="Learning rate"
    )
    args = parser.parse_args()
    finetune_mito(args)


if __name__ == "__main__":
    main()
