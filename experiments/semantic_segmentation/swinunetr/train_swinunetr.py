import sys
import argparse

import torch

import torch_em

from micro_sam.training.semantic_sam_trainer import CustomDiceLoss

from medico_sam.loss import CustomCombinedLoss
from medico_sam.models.monai_models import get_monai_models


sys.path.append("..")


def train_swinunetr(args):
    """Train SwinUNETR model for semantic segmentation on medical imaging datasets.
    """
    from common import get_num_classes, DATASETS_2D, DATASETS_3D, get_dataloaders, get_in_channels

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Some training settings.
    dataset = args.dataset
    num_classes = get_num_classes(dataset)
    in_channels = get_in_channels(dataset)

    if dataset in DATASETS_2D:
        patch_shape = (1024, 1024)
        ndim = 2
    elif dataset in DATASETS_3D:
        patch_shape = (32, 512, 512)
        ndim = 3
    else:
        raise ValueError(f"'{dataset}' is not a valid dataset name or not part of our experiments yet.")

    model = get_monai_models(
        image_size=patch_shape, in_channels=in_channels, out_channels=num_classes, ndim=ndim,
    )

    model.to(device)
    checkpoint_name = f"swinunetr/{dataset}"

    # All stuff we need for training.
    learning_rate = 1e-4
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10}

    train_loader, val_loader = get_dataloaders(
        patch_shape=patch_shape, data_path=args.input_path, dataset_name=dataset, benchmark_models=True,
    )

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
    parser = argparse.ArgumentParser(description="Train SwinUNETR model for the semantic segmentation tasks.")
    parser.add_argument(
        "-d", "--dataset", required=True, help="The name of medical dataset for semantic segmentation."
    )
    parser.add_argument(
        "-i", "--input_path", default="/mnt/vast-nhr/projects/cidas/cca/data",
        help="The filepath to the medical data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--save_root", "-s", default=None,
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run."
    )
    parser.add_argument(
        "--iterations", type=int, default=int(5e4), help="For how many iterations should the model be trained?"
    )
    parser.add_argument(
        "--dice_weight", type=float, default=0.5, help="The weight for dice loss with combined cross entropy loss."
    )
    args = parser.parse_args()
    train_swinunetr(args)


if __name__ == "__main__":
    main()
