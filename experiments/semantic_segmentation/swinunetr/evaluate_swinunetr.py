import sys
from tqdm import tqdm

import numpy as np

import torch

from torch_em.transform.raw import standardize
from torch_em.transform.generic import ResizeLongestSideInputs

from tukra.io import read_image

from monai.inferers import sliding_window_inference

from medico_sam.models.monai_models import get_monai_models
from medico_sam.evaluation.evaluation import calculate_dice_score


sys.path.append("../../evaluation")


DATASETS_2D = [
    # v1 pool of 2d semantic segmentation datasets.
    "oimhs",
    "isic",
    "dca1",
    "cbis_ddsm",
    "piccolo",
    "hil_toothseg",
]

DATASETS_3D = [
    # v1 pool of 3d semantic segmentation datasets.
    "osic_pulmofib",
    "duke_liver",
    "oasis",
    "lgg_mri",
    "leg_3d_us",
    "micro_usp",
    # NEW datasets (I would keep all experiments here on around and report the relevant ones)
    "curvas",
    "amos",
]


def get_num_classes(dataset_name):
    if dataset_name in ["oimhs", "oasis"]:
        num_classes = 5
    elif dataset_name in ["osic_pulmofib", "leg_3d_us", "curvas", "amos"]:
        num_classes = 4
    elif dataset_name in [
        "piccolo", "cbis_ddsm", "dca1", "isic", "hil_toothseg",  # 2d datasets
        "duke_liver", "lgg_mri", "micro_usp",  # 3d datasets
    ]:
        num_classes = 2
    else:
        raise ValueError

    return num_classes


def get_in_channels(dataset):
    if dataset in [
        "hil_toothseg", "cbis_ddsm", "dca1",
        "osic_pulmofib", "leg_3d_us", "micro_usp", "lgg_mri", "duke_liver", "oasis", "curvas", "amos",
    ]:
        return 1
    elif dataset in ["oimhs", "isic", "piccolo"]:
        return 3
    else:
        raise ValueError


@torch.no_grad()
def evaluate_swinunetr(args):
    from semantic_segmentation_2d import get_2d_dataset_paths
    from semantic_segmentation_3d import DATASET_MAPPING_3D, get_3d_dataset_paths

    # Stuff for evaluation.
    dataset = args.dataset
    checkpoint = args.checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset)
    in_channels = get_in_channels(dataset)

    raw_transform = ResizeLongestSideInputs(target_shape=(512, 512))
    label_transform = ResizeLongestSideInputs(target_shape=(512, 512), is_label=True)

    # Get the SwinUNETR model.
    if dataset in DATASETS_2D:
        patch_shape = (1024, 1024)
        ndim = 2
    elif dataset in DATASETS_3D:
        patch_shape = (32, 512, 512)
        tile_shape = patch_shape
        ndim = 3
    else:
        raise ValueError(f"'{dataset}' is not a valid dataset name or not part of our experiments yet.")

    model = get_monai_models(
        image_size=patch_shape, in_channels=in_channels, out_channels=num_classes, ndim=ndim,
    )
    model_state = torch.load(checkpoint)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    if dataset in DATASET_MAPPING_3D:
        image_paths, gt_paths, semantic_class_maps = get_3d_dataset_paths(dataset_name=dataset)
    else:
        image_paths, gt_paths, semantic_class_maps = get_2d_dataset_paths(dataset_name=dataset)

    metrics = {}
    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), desc="Run SwinUNETR predictions", total=len(image_paths)
    ):
        image = read_image(image_path)
        gt = read_image(gt_path)

        # breakpoint()

        if dataset in ["oimhs", "isic", "piccolo"]:  # these are RGB images.
            image = image.transpose(2, 0, 1)
        elif dataset in ["duke_liver", "osic_pulmofib", "micro_usp"]:
            image = image.transpose(2, 0, 1)
            gt = gt.transpose(2, 0, 1)

        # Resize the image
        image = raw_transform(image)
        _ = label_transform(gt)

        # Normalize the image
        image = standardize(image)

        # And prepare them as tensors
        image = torch.from_numpy(image)[None].to(device)

        if image.ndim == 3:
            image = image[None]

        if dataset in DATASETS_3D and image.ndim < 5:
            image = image[None]

        # Get predictions
        if dataset in DATASETS_2D:
            outputs = model(image)
        else:
            outputs = sliding_window_inference(image, tile_shape, 4, model)

        outputs = torch.argmax(outputs, dim=1)
        outputs = outputs.detach().cpu().numpy().squeeze()
        outputs = label_transform.convert_transformed_inputs_to_original_shape(outputs)

        # Evaluate the predictions compared to ground-truth masks.
        for cname, cid in semantic_class_maps.items():
            if cid not in gt:  # If the class is not present in labels, no reason to evaluate on it.
                continue

            score = calculate_dice_score(
                input_=(outputs == cid).astype("uint8"), target=(gt == cid).astype("uint8")
            )

            if cname in metrics:
                metrics[cname].append(score)
            else:
                metrics[cname] = [score]

    # Let's get the mean per class.
    mean_metrics = [float(np.mean(metrics[cname])) for cname in semantic_class_maps.keys()]
    print(mean_metrics)


if __name__ == "__main__":
    from util import get_default_arguments
    args = get_default_arguments()
    evaluate_swinunetr(args)
