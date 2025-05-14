"""Scripts to run inference on models trained in 'joint-training' style (kind of similar to micro-sam).
"""

from tqdm import tqdm

import torch

from torch_em.data.datasets.medical import curvas

from tukra.io import read_image

from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import get_decoder

from medico_sam.util import get_medico_sam_model
from medico_sam.transform import RawTransformJointTraining
from medico_sam.evaluation.inference import _run_semantic_segmentation_for_image_3d


def run_curvas_inference(output_channels, threshold=0.5):
    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/curvas/"
    checkpoint_path = "/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam/joint-training/checkpoints/vit_b/curvas_sam/best.pt"  # noqa

    # Get SAM model.
    predictor, state = get_sam_model(model_type="vit_b", checkpoint_path=checkpoint_path, return_state=True)

    # Get the UNETR decoder.
    decoder = get_decoder(
        image_encoder=predictor.model.image_encoder,
        decoder_state=state["decoder_state"],
        out_channels=1,
        flexible_load_checkpoint=False,
    )

    # Get medico-sam model.
    model = get_medico_sam_model(
        model_type="vit_b",
        checkpoint_path=checkpoint_path,
        device="cuda",
        use_sam3d=True,
        image_size=512,
        decoder=decoder,
    )

    # Get the images and iterative over each!
    image_paths, gt_paths = curvas.get_curvas_paths(path=data_dir, split="test")
    for image_path, gt_path in zip(image_paths, gt_paths):

        # Load images in expected format.
        image = read_image(image_path).transpose(2, 0, 1)

        # Normalize inputs
        raw_transform = RawTransformJointTraining(modality="CT")
        image = raw_transform(image)

        # Get predictions.
        outputs = _run_semantic_segmentation_for_image_3d(
            model=model,
            image=image,
            prediction_path=None,
            patch_shape=(32, 512, 512),
            halo=(8, 0, 0),
        )

        breakpoint()


def main():
    run_curvas_inference(output_channels=1)  # zero-shot binary (all-class) segmentation.


if __name__ == "__main__":
    main()
