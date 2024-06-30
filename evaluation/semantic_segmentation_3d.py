from torch_em.util import load_model

from medico_sam.util import get_medico_sam_model
from medico_sam.evaluation import inference


def transform_labels(y):
    return (y > 0).astype("float32")


def check_lucchi():
    ckpt = "/home/nimcpape/Work/my_projects/medico-sam/semantic_segmentation/checkpoints/lucchi_3d_adapter_lora4"
    model = get_medico_sam_model("vit_b", device="cuda", use_sam3d=True, lora_rank=4, n_classes=2, image_size=512)
    model = load_model(ckpt, device="cuda", model=model)

    input_paths = [
        "/home/nimcpape/Work/my_projects/medico-sam/semantic_segmentation/data/lucchi_test.h5"
    ]
    output_dir = "./pred_lucchi"
    inference.run_semantic_segmentation_3d(
        model, input_paths, output_dir, semantic_class_map={"blub": 0}, is_multiclass=True,
        image_key="raw",
    )


def main():
    check_lucchi()


if __name__ == "__main__":
    main()
