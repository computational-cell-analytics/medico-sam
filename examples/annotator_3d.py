import os

from tukra.io import read_image

from micro_sam.sam_annotator import annotator_3d

from medico_sam.util import get_cache_directory
from medico_sam.sample_data import fetch_ct_example_data


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def osic_pulmofib_annotator(use_finetuned_model):
    """Run the 3d annotator for an example image from the OSIC PulmoFib dataset.

    See https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data for details on the data.
    """
    example_data = fetch_ct_example_data(DATA_CACHE)
    image = read_image(example_data)

    if use_finetuned_model:
        model_type = "vit_b_medical_imaging"
    else:
        model_type = "vit_b"

    annotator_3d(image=image, model_type=model_type)


def main():
    # Whether to use the fine-tuned SAM model for medical imaging data.
    use_finetuned_model = True

    # 2d annotator for osic-pulmofib data
    osic_pulmofib_annotator(use_finetuned_model)


if __name__ == "__main__":
    main()
