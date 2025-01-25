import os

from micro_sam.sam_annotator import image_folder_annotator

from medico_sam.util import get_cache_directory
from medico_sam.sample_data import fetch_fundus_example_data


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def papila_annotator(use_finetuned_model):
    """Run the 3d annotator for an example image from the Papila dataset.

    See https://doi.org/10.1038/s41597-022-01388-1 for details on the data.
    """
    example_data = fetch_fundus_example_data(DATA_CACHE)

    if use_finetuned_model:
        model_type = "vit_b_medical_imaging"
    else:
        model_type = "vit_b"

    image_folder_annotator(
        input_folder=example_data,
        output_folder=os.path.join(get_cache_directory(), "image_series_output_folder"),
        pattern="*.tif",
        model_type=model_type,
    )


def main():
    # Whether to use the fine-tuned SAM model for medical imaging data.
    use_finetuned_model = True

    # image series annotator for papila data
    papila_annotator(use_finetuned_model)


if __name__ == "__main__":
    main()
