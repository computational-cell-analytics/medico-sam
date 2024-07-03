from medico_sam.evaluation.inference import run_semantic_segmentation_3d
from micro_sam.models.sam_3d_wrapper import get_sam_3d_model
from torch_em.util import load_model


def transform_labels(y):
    return (y > 0).astype("float32")


def main():
    # Need to adapt n_classes, lora-rank etc. to your model.
    model = get_sam_3d_model(
        device="cuda", n_classes=2, image_size=512, lora_rank=4
    )
    checkpoint_path = "/home/nimcpape/Work/my_projects/medico-sam/semantic_segmentation/checkpoints/lucchi_3d_adapter_lora4"
    model = load_model(checkpoint_path, model=model, device="cuda")

    image_paths = ["../semantic_segmentation/data/lucchi_test.h5"]
    prediction_dir = "./pred_lucchi"

    run_semantic_segmentation_3d(
        model, image_paths, prediction_dir, semantic_class_map={"dummy": 0},
        image_key="raw"
    )


main()
