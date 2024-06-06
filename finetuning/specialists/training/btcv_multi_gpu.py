import argparse

import torch

from torch_em.loss.dice import BCEDiceLossWithLogits
from torch_em.multi_gpu_training import train_multi_gpu
from torch_em.data.datasets.medical import get_btcv_dataset

import micro_sam.training as sam_training


def finetune_btcv(args):
    """Code for finetuning SAM on BTCV using multiple GPUs in "micro_sam"-based MedSAM reimplementation"""

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (1, 512, 512)  # the patch shape for training
    n_objects_per_batch = args.n_objects  # this is the number of objects per batch that will be sampled (default: 25)
    freeze_parts = args.freeze  # override this to freeze different parts of the model

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts
    )

    # dataset and respective kwargs
    raw_transform = sam_training.identity

    train_dataset_class = get_btcv_dataset
    val_dataset_class = get_btcv_dataset
    train_dataset_kwargs = {
        "path": args.input_path, "patch_shape": patch_shape, "ndim": 2, "raw_transform": raw_transform,
    }
    val_dataset_kwargs = {
        "path": args.input_path, "patch_shape": patch_shape, "ndim": 2, "raw_transform": raw_transform,
    }
    loader_kwargs = {"batch_size": 4, "shuffle": True, "num_workers": 16}

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = sam_training.ConvertToSamInputs(transform=model.transform, box_distortion_factor=0.025)

    checkpoint_name = f"{args.model_type}/btcv_medsam"

    train_multi_gpu(
        model_callable=sam_training.get_trainable_sam_model,
        model_kwargs={"model_type": model_type, "checkpoint_path": checkpoint_path, "freeze": freeze_parts},
        train_dataset_callable=train_dataset_class,
        train_dataset_kwargs=train_dataset_kwargs,
        val_dataset_callable=val_dataset_class,
        val_dataset_kwargs=val_dataset_kwargs,
        loader_kwargs=loader_kwargs,
        iterations=args.iterations,
        optimizer_callable=torch.optim.Adam,
        optimizer_kwargs={"lr": 5e-5},
        lr_scheduler_callable=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs={"mode": "min", "factor": 0.9, "patience": 10, "verbose": True},
        # trainer params
        trainer_callable=sam_training.SimpleSamTrainer,
        name=checkpoint_name,
        save_root=args.save_root,
        logger=sam_training.SamLogger,
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        compile_model=False,
        n_sub_iteration=1,
        mask_prob=0,
        mask_loss=BCEDiceLossWithLogits(),
        use_box=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the BTCV dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/projects/nim00007/data/btcv/",
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
        "--iterations", type=int, default=int(1e4),
        help="For how many iterations should the model be trained?"
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
        "--save_every_kth_epoch", type=int, default=None,
        help="To save every kth epoch while fine-tuning. Expects an integer value."
    )
    parser.add_argument(
        "--n_objects", type=int, default=25, help="The number of instances (objects) per batch used for finetuning."
    )
    args = parser.parse_args()
    finetune_btcv(args)


if __name__ == "__main__":
    main()
