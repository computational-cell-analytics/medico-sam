import os
from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple

import numpy as np
from skimage.measure import label as connected_components

import torch
import torch.nn as nn

from torch_em.transform.raw import normalize
from torch_em.util.segmentation import size_filter
from torch_em.util.prediction import predict_with_halo
from torch_em.transform.generic import ResizeLongestSideInputs

from micro_sam.evaluation.inference import _run_inference_with_iterative_prompting_for_image

from tukra.io import read_image, write_image

from segment_anything import SamPredictor


def run_inference_with_iterative_prompting_per_semantic_class(
    predictor: SamPredictor,
    image_paths: List[Union[str, os.PathLike]],
    gt_paths: List[Union[str, os.PathLike]],
    prediction_dir: Union[str, os.PathLike],
    start_with_box_prompt: bool,
    semantic_class_map: Dict[str, int],
    embedding_dir: Optional[Union[str, os.PathLike]] = None,
    dilation: int = 5,
    batch_size: int = 32,
    n_iterations: int = 8,
    use_masks: bool = False,
    min_size: int = 20,
) -> None:
    """Run segment anything inference for multiple images using prompts iteratively
    derived from model outputs and groundtruth (per semantic class)

    Args:
        predictor: The SegmentAnything predictor.
        image_paths: The image file paths.
        gt_paths: The ground-truth segmentation file paths.
        embedding_dir: The directory where the image embeddings will be saved or are already saved.
        prediction_dir: The directory where the predictions from SegmentAnything will be saved per iteration.
        start_with_box_prompt: Whether to use the first prompt as bounding box or a single point
        dilation: The dilation factor for the radius around the ground-truth object
            around which points will not be sampled.
        batch_size: The batch size used for batched predictions.
        n_iterations: The number of iterations for iterative prompting.
        use_masks: Whether to make use of logits from previous prompt-based segmentation.
    """
    if len(image_paths) != len(gt_paths):
        raise ValueError(f"Expect same number of images and gt images, got {len(image_paths)}, {len(gt_paths)}")

    if use_masks:
        print("The iterative prompting will make use of logits masks from previous iterations.")

    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths),
        desc="Run inference with iterative prompting for all images",
    ):
        image_name = os.path.basename(image_path)

        assert os.path.exists(image_path), image_path
        assert os.path.exists(gt_path), gt_path

        # Perform segmentation only on the semantic class
        for semantic_class_name, semantic_class_id in semantic_class_map.items():
            # We skip the images that already have been segmented
            prediction_paths = [
                os.path.join(
                    prediction_dir, f"iteration{i:02}", semantic_class_name, image_name
                ) for i in range(n_iterations)
            ]
            if all(os.path.exists(prediction_path) for prediction_path in prediction_paths):
                continue

            image = read_image(image_path)
            gt = read_image(gt_path)

            # create all prediction folders for all intermediate iterations
            for i in range(n_iterations):
                os.makedirs(os.path.join(prediction_dir, f"iteration{i:02}", semantic_class_name), exist_ok=True)

            if isinstance(semantic_class_id, int):
                gt = (gt == semantic_class_id).astype("uint32")

            # Once we have the class labels, let's run connected components to label dissociated components, if any.
            # - As an example, this is relevant for aortic structures (etc.), where the aorta could have multiple
            #   branches in the thoracic cavity in the axial view.
            gt = connected_components(gt)

            # Filter out extremely small objects from segmentation
            gt = size_filter(seg=gt, min_size=min_size)

            # Check whether there are objects or it's not relevant for interactive segmentation
            if not len(np.unique(gt)) > 1:
                continue

            if embedding_dir is None:
                embedding_path = None
            else:
                embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(image_name)[0]}.zarr")

            _run_inference_with_iterative_prompting_for_image(
                predictor, image, gt, start_with_box_prompt=start_with_box_prompt,
                dilation=dilation, batch_size=batch_size, embedding_path=embedding_path,
                n_iterations=n_iterations, prediction_paths=prediction_paths, use_masks=use_masks
            )


#
# SEMANTIC SEGMENTATION FUNCTIONALITIES
#


def _run_semantic_segmentation_for_image(
    model: nn.Module,
    image: np.ndarray,
    prediction_path: Union[os.PathLike, str],
):
    # A simple transform to ensure all values are between 0 - 255.
    image = normalize(image) * 255

    if image.ndim == 3:
        image = image.transpose(2, 0, 1)[None]
    else:
        image = image[None, None]

    image = torch.from_numpy(image).to("cuda")  # NOTE: I hard-code the device here because I am lazy.

    outputs = model(image)

    masks = torch.argmax(outputs, dim=1)
    masks = masks.detach().cpu().numpy().squeeze()

    # save the segmentations
    write_image(prediction_path, masks, compression="zlib")


def run_semantic_segmentation(
    model: nn.Module,
    image_paths: List[Union[str, os.PathLike]],
    prediction_dir: Union[str, os.PathLike],
    semantic_class_map: Dict[str, int],
    is_multiclass: bool = False,
):
    """
    """
    for image_path in tqdm(image_paths, desc="Run inference for semantic segmentation with all images"):
        image_name = os.path.basename(image_path)

        assert os.path.exists(image_path), image_path

        # Perform segmentation only on the semantic class
        for i, (semantic_class_name, _) in enumerate(semantic_class_map.items()):
            if is_multiclass:
                semantic_class_name = "all"
                if i > 0:  # We only perform segmentation for multiclass once.
                    continue

            # We skip the images that already have been segmented
            prediction_path = os.path.join(prediction_dir, semantic_class_name, image_name)
            if os.path.exists(prediction_path):
                continue

            image = read_image(image_path)

            # create the prediction folder
            os.makedirs(os.path.join(prediction_dir, semantic_class_name), exist_ok=True)

            _run_semantic_segmentation_for_image(
                model=model, image=image, prediction_path=prediction_path,
            )


@torch.no_grad()
def _run_semantic_segmentation_for_image_3d(
    model: torch.nn.Module,
    image: np.ndarray,
    prediction_path: Optional[Union[os.PathLike, str]],
    patch_shape: Tuple[int, int, int],
    halo: Tuple[int, int, int],
) -> np.ndarray:
    device = next(model.parameters()).device
    block_shape = tuple(bs - 2 * ha for bs, ha in zip(patch_shape, halo))

    def preprocess(x):
        x = 255 * normalize(x)
        x = np.stack([x] * 3, axis=0)
        return x

    # First, we reshape the YX dimension for 3d inputs
    resize_transform = ResizeLongestSideInputs(target_shape=(512, 512))
    image = resize_transform(image)

    # Custom prepared function to infer per tile.
    def prediction_function(net, inp):
        masks = net(inp.squeeze(0))
        masks = torch.argmax(masks, dim=1)
        return masks

    output = np.zeros(image.shape, dtype="float32")
    predict_with_halo(
        input_=image,
        model=model,
        gpu_ids=[device],
        block_shape=block_shape,
        halo=halo,
        preprocess=preprocess,
        output=output,
        prediction_function=prediction_function
    )

    # Lastly, we resize the predictions back to the original shape.
    output = resize_transform.convert_transformed_inputs_to_original_shape(output)

    if prediction_path is not None:
        # save the segmentations
        write_image(prediction_path, output, compression="zlib")

    return output


def run_semantic_segmentation_3d(
    model: torch.nn.Module,
    image_paths: List[Union[str, os.PathLike]],
    prediction_dir: Union[str, os.PathLike],
    semantic_class_map: Dict[str, int],
    patch_shape: Tuple[int, int, int] = (16, 512, 512),
    halo: Tuple[int, int, int] = (4, 0, 0),
    image_key: Optional[str] = None,
    is_multiclass: bool = False,
    make_channels_first: bool = False,
):
    """Run inference for semantic segmentation in 3d.

    Args:
        model: The choice of model for 3d semantic segmentation.
        image_paths: List of filepaths to the image data.
        prediction_dir: Filepath to store predictions.
        semantic_class_map: The map to semantic classes.
        patch_shape: The patch shape used for training 3d semantic segmentation.
        halo: The overlay for tiling window-based prediction.
        image_key: The hierarchy name for container data structures.
        is_multiclass: Whether the semantic segmentation is for multiple classes.
        make_channels_first: Whether to make channels first for inputs.
    """
    for image_path in tqdm(image_paths, desc="Run inference for semantic segmentation with all images"):
        image_name = os.path.basename(image_path)

        assert os.path.exists(image_path), image_path

        # Perform segmentation only on the semantic class
        for i, (semantic_class_name, _) in enumerate(semantic_class_map.items()):
            if is_multiclass:
                semantic_class_name = "all"
                if i > 0:  # We only perform segmentation for multiclass once.
                    continue

            # We skip the images that already have been segmented
            image_name = Path(image_name.split(".")[0]).with_suffix(".tif")
            prediction_path = os.path.join(prediction_dir, semantic_class_name, image_name)
            if os.path.exists(prediction_path):
                continue

            image = read_image(image_path, key=image_key)

            if make_channels_first:
                image = image.transpose(2, 0, 1)

            # create the prediction folder
            os.makedirs(os.path.join(prediction_dir, semantic_class_name), exist_ok=True)

            _run_semantic_segmentation_for_image_3d(
                model=model,
                image=image,
                prediction_path=prediction_path,
                patch_shape=patch_shape,
                halo=halo,
            )
