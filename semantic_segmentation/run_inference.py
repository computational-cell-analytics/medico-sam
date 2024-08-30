import os
import argparse
from glob import glob

import torch
import h5py

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.util import update_kwargs_for_resize_trafo
from torch_em import default_segmentation_dataset
from torch_em.transform.label import BoundaryTransform, NoToBackgroundBoundaryTransform
from torch_em.util import load_model

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model
from micro_sam.models.sam_3d_wrapper import get_sam_3d_model
from micro_sam.training.util import ConvertToSemanticSamInputs
#from micro_sam.evaluation.inference import run_amg


from medico_sam.transform.raw import RawResizeTrafoFor3dInputs
from medico_sam.transform.label import LabelResizeTrafoFor3dInputs
from medico_sam.util import get_medico_sam_model
from medico_sam.evaluation import inference


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

    kwargs["raw_transform"] = RawResizeTrafoFor3dInputs(desired_shape=patch_shape)

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


def run_inference(args):
    """Code for finetuning SAM on Duke Liver for semantic segmentation."""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = args.checkpoint  # override this to start training from a custom checkpoint
    patch_shape = (64, 512, 512)  # the patch shape for training
    halo = (8, 64, 64)
    num_classes = 2  # 1 background class, 1 semantic foreground class
    lora_rank = 8
    path = "/scratch-grete/projects/nim00007/data/mitochondria/cooper/mito_tomo/outer-membrane2/2_20230415_TOMO_HOI_WT_36859_J1_STEM750/36859_J1_STEM750_66K_SP_07_rec_2kb1dawbp_crop.h5"
    #checkpoint_path = "/scratch-grete/usr/nimlufre/medico_sam/mito_segmentation_lrs/1e-4/checkpoints/vit_b_3d_lora4/bs2/"
    ds = get_mito_dataset(path, patch_shape)
    model = get_medico_sam_model(
        model_type="vit_b",
        device=device,
        use_sam3d=True,
        lora_rank=lora_rank,
        n_classes=num_classes,
        image_size=512
    )
    model = load_model(checkpoint_path, device=device, model=model)
    
    image, label = next(iter(ds))
    with h5py.File(path, "r") as f:
        image = f["raw"][:]
    print("iamge shape", image.shape)
    inference._run_semantic_segmentation_for_image_3d(
        model=model,
        image=image,
        prediction_path=args.save_root+"vit_b_3d_lora4-bs2-prediction_of-36859_J1_STEM750_66K_SP_07_rec_2kb1dawbp_crop.tif",
        patch_shape=patch_shape,
        halo=halo
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
        "--save_root", "-s", default="/scratch-grete/usr/nimlufre/medico_sam/mito_segmentation_lrs/",
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
    run_inference(args)


if __name__ == "__main__":
    main()