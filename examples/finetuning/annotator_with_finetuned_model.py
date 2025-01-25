from tukra.io import read_image

from micro_sam.sam_annotator import annotator_2d


def run_annotator_with_finetuned_model():
    """Run the 2d anntator with a custom (finetuned) model.

    Here, we use the model that is produced by `finetuned_sam.py` and apply it
    for an image not included in the training set.
    """
    # take the last frame, which is part of the val set, so the model was not directly trained on it
    im = read_image("./data/PSFHS/image_mha/05000.mha")

    # set the checkpoint and the path for caching the embeddings
    checkpoint = "./finetuned_psfhs_model.pth"

    # Adapt this if you finetune a different model type, e.g. vit_h.
    model_type = "vit_b"  # We finetune a vit_b in the example script.

    # Run the 2d annotator with the custom model.
    annotator_2d(im, model_type=model_type, checkpoint=checkpoint)


if __name__ == "__main__":
    run_annotator_with_finetuned_model()
