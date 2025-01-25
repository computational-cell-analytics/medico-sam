import torch

from torch_em.data.datasets import medical
from torch_em.data import MinInstanceSampler

import micro_sam.training as sam_training
from micro_sam.models import sam_3d_wrapper
from micro_sam.training.util import ConvertToSemanticSamInputs

from medico_sam.util import LinearWarmUpScheduler
from medico_sam.transform.raw import RawResizeTrafoFor3dInputs
from medico_sam.transform.label import LabelResizeTrafoFor3dInputs


DATA_ROOT = "data"


def get_dataloader(split):
    """
    """
    patch_shape = (8, 512, 512)

    loader = medical.get_lgg_mri_loader(
        path=DATA_ROOT,
        batch_size=1,
        split=split,
        channels="flair",
        resize_inputs=True,
        patch_shape=patch_shape,
        num_workers=16,
        shuffle=True,
        pin_memory=True,
        sampler=MinInstanceSampler(),
        raw_transform=RawResizeTrafoFor3dInputs(desired_shape=patch_shape),
        label_transform=LabelResizeTrafoFor3dInputs(desired_shape=patch_shape),
        download=True,
    )
    return loader


def finetune_semantic_sam():
    """Code for finetuning SAM on medical datasets for semantic segmentation."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = "vit_b"
    num_classes = 2  # 1 background class and 'n' semantic foreground classes

    # get the trainable segment anything model
    model = sam_3d_wrapper.get_sam_3d_model(device=device, n_classes=num_classes, image_size=512)
    model.to(device)
    checkpoint_name = f"{model_type}/lgg_mri_semanticsam"

    train_loader = get_dataloader("train")
    val_loader = get_dataloader("val")

    # all the stuff we need for training
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    mscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5)
    scheduler = LinearWarmUpScheduler(optimizer, warmup_epochs=4, main_scheduler=mscheduler)

    train_loader = get_dataloader("train")
    val_loader = get_dataloader("val")

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = ConvertToSemanticSamInputs()

    # the trainer which performs the semantic segmentation training and validation (implemented using "torch_em")
    trainer = sam_training.SemanticSamTrainer(
        name=checkpoint_name,
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
        dice_weight=0.5,
    )
    trainer.fit(epochs=100)


def main():
    finetune_semantic_sam()


if __name__ == "__main__":
    main()
