import argparse

import torch
import micro_sam.training as sam_training
from micro_sam.models.sam_3d_wrapper import get_sam_3d_model
from micro_sam.models.simple_sam_3d_wrapper import get_simple_sam_3d_model
from micro_sam.training.util import ConvertToSemanticSamInputs

from torch_em.data.datasets.electron_microscopy import get_lucchi_loader

from common import RawTrafoFor3dInputs


def transform_labels(y):
    return (y > 0).astype("float32")


def get_loaders(patch_shape):
    train_loader = get_lucchi_loader(
        "./data", split="train", patch_shape=patch_shape, batch_size=1, download=True,
        raw_transform=RawTrafoFor3dInputs(), label_transform=transform_labels,
        n_samples=100
    )
    val_loader = get_lucchi_loader(
        "./data", split="test", patch_shape=patch_shape, batch_size=1,
        raw_transform=RawTrafoFor3dInputs(), label_transform=transform_labels
    )
    return train_loader, val_loader


def train_lucchi(args):
    patch_shape = (32, 512, 512)
    train_loader, val_loader = get_loaders(patch_shape)

    lora_rank = args.lora_rank
    train_simple = args.train_simple

    lora_str = "frozen" if lora_rank is None else f"lora{lora_rank}"
    freeze_encoder = True if lora_rank is None else False
    name = f"lucchi_3d_{'simple' if train_simple else 'adapter'}_{lora_str}"

    model_type = "vit_b"
    device = "cuda"
    num_classes = 2

    model_function = get_simple_sam_3d_model if train_simple else get_sam_3d_model

    # get the trainable segment anything model
    model = model_function(
        model_type=model_type,
        device=device,
        n_classes=num_classes,
        image_size=patch_shape[1],
        checkpoint_path=None,
        freeze_encoder=freeze_encoder,
        lora_rank=lora_rank,
    )
    model.to(device)

    # all the stuff we need for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=3, verbose=True)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = ConvertToSemanticSamInputs()

    # the trainer which performs the semantic segmentation training and validation (implemented using "torch_em")
    trainer = sam_training.semantic_sam_trainer.SemanticSamTrainer(
        name=name,
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
    trainer.fit(int(2.5e4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lora_rank", default=None, type=int)
    parser.add_argument("-s", "--train_simple", action="store_true")

    args = parser.parse_args()
    train_lucchi(args)


if __name__ == "__main__":
    main()
