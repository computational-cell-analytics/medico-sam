import os
from tqdm import tqdm
from typing import List, Union, Dict, Optional

import numpy as np
import imageio.v3 as imageio
from skimage.transform import resize
from skimage.measure import label as connected_components

import torch

from torch_em.util.segmentation import size_filter

from micro_sam import util
from micro_sam.evaluation.inference import _run_inference_with_iterative_prompting_for_image

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
    min_size: int = 0,
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

            image = imageio.imread(image_path)
            gt = imageio.imread(gt_path)

            # create all prediction folders for all intermediate iterations
            for i in range(n_iterations):
                os.makedirs(os.path.join(prediction_dir, f"iteration{i:02}", semantic_class_name), exist_ok=True)

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
    predictor: SamPredictor,
    image,
    embedding_path,
    prediction_path,
    mask_threshold=0.6,
):
    # Compute the image embeddings.
    image_embeddings = util.precompute_image_embeddings(
        predictor, image, embedding_path, ndim=2, verbose=False,
    )
    util.set_precomputed(predictor, image_embeddings)

    # Get the predictions out of the SamPredictor
    batch_masks, batch_ious, batch_logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=None,
        mask_input=None,
        multimask_output=True,
        return_logits=True,
    )

    import matplotlib.pyplot as plt

    #
    # APPROACH 1:
    #
    # masks = torch.softmax(batch_masks, dim=1)
    # masks = torch.argmax(masks, dim=1)
    # masks = masks.detach().cpu().numpy().squeeze()
    # NOTE: below resizing only for batch_logits
    # masks = resize(
    #     image=image,
    #     output_shape=image.shape[:2],
    #     preserve_range=True,
    #     order=0,
    #     anti_aliasing=False
    # )
    # cols = 1 + masks.shape[0] if masks.ndim == 3 else 2
    # fig, ax = plt.subplots(1, cols, figsize=(20, 20))
    # ax[0].imshow(image.astype("uint8"))

    # if masks.ndim == 2:
    #     masks = masks[None]
    # for i, mask in enumerate(masks, start=1):
    #     ax[i].imshow(mask)

    # plt.savefig("./seg.png")
    # plt.close()

    #
    # APPROACH 2:
    #
    masks = torch.sigmoid(batch_masks)
    masks = masks.detach().cpu().numpy().squeeze()

    fig, ax = plt.subplots(1, 5, figsize=(20, 20))
    ax[0].imshow(image.astype("uint8"))
    for i in range(masks.shape[0]):
        ax[i+1].imshow(masks[i] > 0.6)

    plt.savefig("./seg.png")
    plt.close()

    breakpoint()

    # save the segmentations
    # imageio.imwrite(prediction_path, masks, compression="zlib")


def run_semantic_segmentation(
    predictor: SamPredictor,
    image_paths: List[Union[str, os.PathLike]],
    prediction_dir: Union[str, os.PathLike],
    semantic_class_map: Dict[str, int],
    embedding_dir: Optional[Union[str, os.PathLike]] = None,
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

            image = imageio.imread(image_path)

            # create the prediction folder
            os.makedirs(os.path.join(prediction_dir, semantic_class_name), exist_ok=True)

            if embedding_dir is None:
                embedding_path = None
            else:
                embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(image_name)[0]}.zarr")

            _run_semantic_segmentation_for_image(
                predictor=predictor, image=image, embedding_path=embedding_path, prediction_path=prediction_path,
            )
