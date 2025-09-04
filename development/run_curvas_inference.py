"""Scripts to run inference on models trained in 'joint-training' style (kind of similar to micro-sam).

Downstream multi-class semantic segmentation (2d):
- CURVAS (w decoder init): 0.7819717042043587 (time: 2.74h)
- CURVAS (w/o decoder init): 0.7233613322122812 (time: 2.03h)

- AMOS (w decoder init): 0.7927464267441214 (time: 20.39h)
- AMOS (w/o decoder init): 0.7969678391504248 (time: 22.51h)
"""

import os
from tqdm import tqdm

import numpy as np

import torch

from torch_em.util import load_model
from torch_em.data.datasets import medical
from torch_em.transform.raw import normalize
from torch_em.transform.generic import ResizeLongestSideInputs

from tukra.io import read_image

from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import get_unetr

from medico_sam.util import get_medico_sam_model
from medico_sam.evaluation.evaluation import calculate_dice_score
from medico_sam.evaluation.inference import _run_semantic_segmentation_for_image_3d


def custom_raw_transform(raw):
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


def get_model(output_channels, checkpoint_path, simple_unetr=True):
    # Get SAM model.
    predictor = get_sam_model(model_type="vit_b")

    if simple_unetr:
        # Get the 2d UNETR model.
        model = get_unetr(image_encoder=predictor.model.image_encoder, out_channels=output_channels)

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
        input_paths = medical.curvas.get_curvas_paths(path=os.path.join(data_dir, "curvas"), split="test")
        image_paths = gt_paths = input_paths

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


def run_curvas_inference(dataset, checkpoint, output_channels):
    # Stuff for running inference properly.
    simple_unetr = True  # whether to use 2d unet or sam+unetr volumetric setup.

    # Get the semantic segmentation model
    model = get_model(output_channels, checkpoint, simple_unetr=simple_unetr)

    # Get the images and iterative over each!
    image_paths, gt_paths = get_dataset_paths(dataset_name=dataset)
    scores = []
    for i, (image_path, gt_path) in enumerate(zip(image_paths, gt_paths)):

        # Load images in expected format.
        if dataset == "curvas":
            image = read_image(image_path, key="raw")
            gt = read_image(gt_path, key="labels/rater_1")
        else:
            image = read_image(image_path).transpose(2, 0, 1)
            gt = read_image(gt_path).transpose(2, 0, 1)

        # Get binary labels.
        if dataset == "curvas":
            gt = (gt > 0).astype("uint8")  # Binarize labels.
        elif dataset == "amos":
            raw_transform = ResizeLongestSideInputs(target_shape=(512, 512))
            image = raw_transform(image)

            label_transform = ResizeLongestSideInputs(target_shape=(512, 512), is_label=True)
            gt = label_transform(gt)
            # gt = np.isin(gt, [2, 3, 6, 10]).astype("uint8")  # valid class ids we want for this task.

        # Get predictions.
        if simple_unetr:
            # Normalize images per slice.
            image = np.stack([custom_raw_transform(im) for im in image])

            image_tensor = torch.from_numpy(image).to("cuda")
            outputs = [
                model(per_slice[None, None]).detach().cpu().squeeze()
                for per_slice in tqdm(image_tensor, desc="Run predictions")
            ]
            outputs = torch.stack(outputs, dim=0)
            outputs = torch.argmax(outputs, dim=1)

        else:
            outputs = _run_semantic_segmentation_for_image_3d(
                model=model,
                image=image,
                prediction_path=None,
                patch_shape=(16, 512, 512),
                halo=(4, 0, 0),
            )

        # HACK
        gt = filter_valid_labels(gt)
        outputs = outputs.numpy().astype("uint8")

        # Evaluate predictions compared to the ground-truth
        for id in [1, 2, 3]:
            dice_score = calculate_dice_score(input_=(gt == id), target=(outputs == id))
            scores.append(dice_score)
            print(dice_score)

        save_results = True
        if save_results:
            import h5py
            with h5py.File(f"results_{os.path.basename(os.path.dirname(checkpoint))}_{i}.h5", "w") as f:
                f.create_dataset("raw", data=image, compression="gzip")
                f.create_dataset("labels", data=gt, compression="gzip")
                f.create_dataset("prediction", data=outputs, compression="gzip")

    # print(np.mean(scores))


def main(args):
    # zero-shot binary (all-class) segmentation.
    # checkpoint_path = "/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam/joint-training/checkpoints/vit_b/curvas_sam/best.pt"  # noqa
    # run_curvas_inference("curvas", checkpoint_path, output_channels=1)

    # multi-class 2d segmentation.
    run_curvas_inference(dataset=args.dataset, checkpoint=args.checkpoint, output_channels=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Name of the chosen dataset",
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, required=True, help="Filepath to trained model checkpoint."
    )
    args = parser.parse_args()
    main(args)
