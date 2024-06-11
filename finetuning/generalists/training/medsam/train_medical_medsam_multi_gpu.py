import argparse

import torch

from torch_em.data import MinInstanceSampler
from torch_em.multi_gpu_training import train_multi_gpu
from torch_em.data.datasets.medical import get_sa_med2d_dataset

import micro_sam.training as sam_training

from segment_anything.utils.transforms import ResizeLongestSide


def finetune_medical_generalist(args):
    """Code for finetuning SAM in MedSAM-style on SA-Med2D-20M dataset using multiple GPUs,
    composed of multiple medical datasets
    """
    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (1024, 1024)  # the patch shape for training
    n_objects_per_batch = args.n_objects  # this is the number of objects per batch that will be sampled (default: 25)
    freeze_parts = args.freeze  # override this to freeze one or more of these backbones
    checkpoint_name = f"{args.model_type}/medical_generalist_medsam_multi_gpu"

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = sam_training.ConvertToSamInputs(
        transform=ResizeLongestSide(target_length=1024), box_distortion_factor=0.025
    )

    # dataset and respective kwargs
    raw_transform = sam_training.identity

    train_dataset_class = get_sa_med2d_dataset
    val_dataset_class = get_sa_med2d_dataset
    train_dataset_kwargs = {
        "path": args.input_path,
        "patch_shape": patch_shape,
        "split": "train",
        "resize_inputs": True,
        "raw_transform": raw_transform,
        "sampler": MinInstanceSampler(),
        "n_fraction_per_dataset": 0.5,  # training on 50% of the train-split
    }
    val_dataset_kwargs = {
        "path": args.input_path,
        "patch_shape": patch_shape,
        "split": "val",
        "resize_inputs": True,
        "raw_transform": raw_transform,
        "sampler": MinInstanceSampler(),
        "n_fraction_per_dataset": 0.1,  # validating on 10% of the val-split
    }

    loader_kwargs = {
        "batch_size": 7,
        "shuffle": True,
        "num_workers": 16,
        "pin_memory": True
    }

    train_multi_gpu(
        model_callable=sam_training.get_trainable_sam_model,
        model_kwargs={"model_type": model_type, "checkpoint_path": checkpoint_path, "freeze": freeze_parts},
        train_dataset_callable=train_dataset_class,
        train_dataset_kwargs=train_dataset_kwargs,
        val_dataset_callable=val_dataset_class,
        val_dataset_kwargs=val_dataset_kwargs,
        loader_kwargs=loader_kwargs,
        iterations=args.iterations,
        find_unused_parameters=True,
        optimizer_callable=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-4},
        # (optional): the original implementation uses no LR scheduler, hence we set the step-size as the last epoch
        lr_scheduler_callable=torch.optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs={"step_size": 10, "gamma": 0.9, "verbose": True},
        # trainer params
        trainer_callable=sam_training.MedSAMTrainer,
        name=checkpoint_name,
        save_root=args.save_root,
        logger=sam_training.SamLogger,
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        compile_model=False,
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
        "--freeze", type=str, nargs="+", default=None,
        help="Which parts of the model to freeze for finetuning."
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
