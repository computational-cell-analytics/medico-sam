"""Scripts to run inference on models trained in 'joint-training' style (kind of similar to micro-sam).
"""

import os
from tqdm import tqdm

import numpy as np

import torch

from torch_em.util import load_model
from torch_em.data.datasets import medical
from torch_em.transform.generic import ResizeLongestSideInputs

from tukra.io import read_image

from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import get_unetr

from medico_sam import transform
from medico_sam.util import get_medico_sam_model
from medico_sam.evaluation.evaluation import calculate_dice_score
from medico_sam.evaluation.inference import _run_semantic_segmentation_for_image_3d


def get_model(output_channels, checkpoint_path, simple_unetr=True):
    # Get SAM model.
    predictor, state = get_sam_model(model_type="vit_b", return_state=True)

    if simple_unetr:
        # Get the 2d UNETR model.
        model = get_unetr(
            image_encoder=predictor.model.image_encoder,
            decoder_state=state["decoder_state"],
            out_channels=output_channels,
        )

    else:
        # Get the 3d medico-sam model with UNETR decoder.
        model = get_medico_sam_model(
            model_type="vit_b",
            device="cuda",
            use_sam3d=True,  # Selecting the 3d model.
            image_size=512,
            decoder_choice="unetr",  # Choose the UNETR decoder
            n_classes=output_channels,
        )
        model = load_model(checkpoint=checkpoint_path, model=model, device="cuda")

    model.eval()
    model.to("cuda")

    return model


def get_dataset_paths(dataset_name):
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data"

    if dataset_name == "curvas":
        image_paths, gt_paths = medical.curvas.get_curvas_paths(path=os.path.join(data_dir, "curvas"), split="test")

    elif dataset_name == "amos":
        # NOTE: right kidney: 2, left kidney: 3, liver: 6, pancreas: 10 (ids for the relevant classes)
        image_paths, gt_paths = medical.amos.get_amos_paths(
            path=os.path.join(data_dir, "amos"), split="val", modality="CT",
        )
        # HACK: use the first 10 images for trying out stuff
        image_paths, gt_paths = image_paths[:10], gt_paths[:10]

    else:
        raise ValueError

    return image_paths, gt_paths


def run_curvas_inference(output_channels):
    # Stuff for running inference properly.
    simple_unetr = False  # whether to use 2d unet or sam+unetr volumetric setup.
    dataset_name = "amos"  # curvas / amos

    # CURVAS (joint training)
    # checkpoint_path = "/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam/joint-training/checkpoints/vit_b/curvas_sam/best.pt"  # noqa
    # CURVAS (semantic segmentation).
    # checkpoint_path = "/mnt/vast-nhr/home/nimanwai/medico-sam/experiments/semantic_segmentation/experiment_curvas/checkpoints/vit_b_3d_all/curvas_semanticsam/best.pt"  # noqa
    # AMOS (semantic segmentation).
    checkpoint_path = "/mnt/vast-nhr/home/nimanwai/medico-sam/experiments/semantic_segmentation/experiment_amos/checkpoints/vit_b_3d_all/amos_semanticsam/best.pt"  # noqa

    # Get the semantic segmentation model
    model = get_model(output_channels, checkpoint_path, simple_unetr=simple_unetr)

    # Get the images and iterative over each!
    image_paths, gt_paths = get_dataset_paths(dataset_name=dataset_name)
    for i, (image_path, gt_path) in enumerate(zip(image_paths, gt_paths)):

        # Load images in expected format.
        image = read_image(image_path).transpose(2, 0, 1)
        gt = read_image(gt_path).transpose(2, 0, 1)

        # Normalize inputs
        raw_transform = transform.RawTransformJointTraining(modality="CT")
        image = raw_transform(image)

        # Get binary labels.
        if dataset_name == "curvas":
            gt = (gt > 0).astype("uint8")  # Binarize labels.
        elif dataset_name == "amos":
            raw_transform = ResizeLongestSideInputs(target_shape=(512, 512))
            image = raw_transform(image)

            label_transform = ResizeLongestSideInputs(target_shape=(512, 512), is_label=True)
            gt = label_transform(gt)
            gt = np.isin(gt, [2, 3, 6, 10]).astype("uint8")  # valid class ids we want for this task.

        # Get predictions.
        if simple_unetr:
            image_tensor = torch.from_numpy(image).to("cuda")
            outputs = [
                model(per_slice[None, None]).detach().cpu().squeeze()
                for per_slice in tqdm(image_tensor, desc="Run predictions")
            ]
            outputs = torch.stack(outputs, axis=0)
            outputs = (outputs.numpy() > 0.5).astype("uint8")  # setting a threshold at 0.5

        else:
            outputs = _run_semantic_segmentation_for_image_3d(
                model=model,
                image=image,
                prediction_path=None,
                patch_shape=(16, 512, 512),
                halo=(4, 0, 0),
            )

        # Evaluate predictions compared to the ground-truth
        dice_score = calculate_dice_score(input_=gt, target=outputs)
        print(dice_score)

        save_results = True
        if save_results:
            import h5py
            with h5py.File(f"results_{i}.h5", "w") as f:
                f.create_dataset("raw", data=image, compression="gzip")
                f.create_dataset("labels", data=gt, compression="gzip")
                f.create_dataset("prediction", data=outputs, compression="gzip")


def main():
    # run_curvas_inference(output_channels=1)  # zero-shot binary (all-class) segmentation.
    run_curvas_inference(output_channels=4)  # zero-shot binary (all-class) segmentation.


if __name__ == "__main__":
    main()
