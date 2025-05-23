import os

import numpy as np

import torch

import torch_em
from torch_em.data.datasets import medical
from torch_em.transform.raw import normalize
from torch_em.data import MinInstanceSampler, MinSemanticLabelForegroundSampler

from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import get_unetr
from micro_sam.training.semantic_sam_trainer import CustomDiceLoss


ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
SAVE_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam/downstream_semantic_segmentation"


def raw_transform(raw):
    # TODO: For CT, use z-score normalization or something better than this!
    raw = normalize(raw)
    raw = raw * 255
    return raw


def filter_valid_labels(labels):
    out = np.zeros_like(labels)

    out[(labels == 2) | (labels == 3)] = 1  # Merge and map kidneys to one id.
    out[labels == 6] = 2  # Map liver id
    out[labels == 10] = 3  # Map pancreas id

    return out


class CustomCombinedLoss(torch.nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 0.5):
        super().__init__()

        self.dice_weight = dice_weight
        self.device = "cuda"

        self.dice_loss = CustomDiceLoss(num_classes=num_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, pred, target):
        pred = pred.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        # Compute the dice loss.
        dice_loss = self.dice_loss(pred, target)

        # Compute cross entropy loss.
        ce_loss = self.ce_loss(pred, target.squeeze(1).long())

        # Get the overall computed loss.
        net_loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * ce_loss

        return net_loss


def get_dataloaders(dataset_name, split):
    if dataset_name == "curvas":
        loader = medical.get_curvas_loader(
            path=os.path.join(ROOT, "curvas"),
            batch_size=8 if split == "train" else 1,
            patch_shape=(1, 512, 512),
            ndim=2,
            resize_inputs=True,
            raw_transform=raw_transform,
            sampler=MinInstanceSampler(min_num_instances=4),
            split=split,
            shuffle=True,
            n_samples=800,
            num_workers=16,
            download=True,
        )

    elif dataset_name == "amos":
        loader = medical.get_amos_loader(
            path=os.path.join(ROOT, "amos"),
            batch_size=8 if split == "train" else 1,
            patch_shape=(1, 512, 512),
            ndim=2,
            resize_inputs=True,
            raw_transform=raw_transform,
            label_transform=filter_valid_labels,
            sampler=MinSemanticLabelForegroundSampler(semantic_ids=[2, 3, 6, 10], min_fraction=25),
            split=split,
            shuffle=True,
            num_workers=16,
        )

    else:
        raise NotImplementedError

    return loader


def train_semantic_2d(dataset_name, checkpoint_path, init_decoder):

    train_loader = get_dataloaders(dataset_name, "train")
    val_loader = get_dataloaders(dataset_name, "val")
    n_classes = 4  # The 4 channels stay consistent for both datasets.

    # Get a simple 2d unetr model.
    predictor, state = get_sam_model(model_type="vit_b", checkpoint_path=checkpoint_path, return_state=True)
    decoder_state = state.get("decoder_state", None)

    # We remove `out_conv`-related parameters and let it initialize from scratch.
    if decoder_state:
        for k in list(state["decoder_state"].keys()):
            if k.startswith("out_conv"):
                del decoder_state[k]

    model = get_unetr(
        image_encoder=predictor.model.image_encoder,
        decoder_state=decoder_state if init_decoder else None,
        out_channels=n_classes,
        flexible_load_checkpoint=True,
    )

    # All the stuff we need for training
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 5}

    # And the trainer for semantic segmentation.
    trainer = torch_em.default_segmentation_trainer(
        name=(dataset_name + ("_with_decoder" if init_decoder else "_without_decoder")),
        save_root=SAVE_ROOT,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device="cuda",
        learning_rate=1e-5,
        scheduler_kwargs=scheduler_kwargs,
        log_image_interval=100,
        mixed_precision=True,
        compile_model=False,
        loss=CustomCombinedLoss(num_classes=n_classes),
        metric=CustomDiceLoss(num_classes=n_classes),
    )
    trainer.fit(iterations=int(5e4), overwrite_training=False)


def main(args):
    # Filepath to the model trained with joint training.
    checkpoint_path = "/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam/joint-training/checkpoints/vit_b/curvas_sam/best.pt"  # noqa

    train_semantic_2d(
        dataset_name=args.dataset, checkpoint_path=checkpoint_path, init_decoder=args.init_decoder
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Name of the chosen dataset",
    )
    parser.add_argument(
        "--init_decoder", action="store_true", help="Whether to initialize the decoder with pretrained weights."
    )
    args = parser.parse_args()
    main(args)
