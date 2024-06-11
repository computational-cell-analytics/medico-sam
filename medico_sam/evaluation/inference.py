import os
from tqdm import tqdm
from typing import List, Union, Dict

import imageio.v3 as imageio

from micro_sam.evaluation.inference import _run_inference_with_iterative_prompting_for_image

from segment_anything import SamPredictor


def run_inference_with_iterative_prompting_per_semantic_class(
    predictor: SamPredictor,
    image_paths: List[Union[str, os.PathLike]],
    gt_paths: List[Union[str, os.PathLike]],
    embedding_dir: Union[str, os.PathLike],
    prediction_dir: Union[str, os.PathLike],
    start_with_box_prompt: bool,
    semantic_class_map: Dict[str, int],
    dilation: int = 5,
    batch_size: int = 32,
    n_iterations: int = 8,
    use_masks: bool = False
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

    # create all prediction folders for all intermediate iterations
    for i in range(n_iterations):
        os.makedirs(os.path.join(prediction_dir, f"iteration{i:02}"), exist_ok=True)

    if use_masks:
        print("The iterative prompting will make use of logits masks from previous iterations.")

    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths),
        total=len(image_paths),
        desc="Run inference with iterative prompting for all images",
    ):
        image_name = os.path.basename(image_path)

        assert os.path.exists(image_path), image_path
        assert os.path.exists(gt_path), gt_path

        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)

        # Perform segmentation only on the semantic class
        for semantic_class_name, semantic_class_id in semantic_class_map.items():
            gt = (gt == semantic_class_id).astype("uint32")

            # We skip the images that already have been segmented
            prediction_paths = [
                os.path.join(
                    prediction_dir, f"iteration{i:02}", semantic_class_name, image_name
                ) for i in range(n_iterations)
            ]
            if all(os.path.exists(prediction_path) for prediction_path in prediction_paths):
                continue

            embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(image_name)[0]}.zarr")

            _run_inference_with_iterative_prompting_for_image(
                predictor, image, gt, start_with_box_prompt=start_with_box_prompt,
                dilation=dilation, batch_size=batch_size, embedding_path=embedding_path,
                n_iterations=n_iterations, prediction_paths=prediction_paths, use_masks=use_masks
            )
