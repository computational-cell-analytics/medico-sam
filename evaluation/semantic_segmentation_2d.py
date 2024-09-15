import os

from medico_sam.evaluation import inference
from medico_sam.util import get_medico_sam_model
from medico_sam.evaluation.evaluation import run_evaluation_for_semantic_segmentation

from util import get_dataset_paths, get_default_arguments, _clear_files, MULTICLASS_SEMANTIC


def _run_semantic_segmentation(image_paths, semantic_class_maps, exp_folder, predictor, is_multiclass):
    prediction_root = os.path.join(exp_folder, "semantic_segmentation")
    embedding_folder = None  # HACK: compute embeddings on-the-fly now, else: os.path.join(exp_folder, "embeddings")
    inference.run_semantic_segmentation(
        predictor=predictor,
        image_paths=image_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_root,
        semantic_class_map=semantic_class_maps,
        is_multiclass=is_multiclass,
    )
    return prediction_root


def main():
    args = get_default_arguments()

    image_paths, gt_paths, semantic_class_maps = get_dataset_paths(dataset_name=args.dataset, split="test")

    # get the predictor to perform inference
    predictor = get_medico_sam_model(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        flexible_load_checkpoint=True,
        num_multimask_outputs=(len(semantic_class_maps.keys()) + 1),
        lora_rank=args.lora_rank,
    )

    prediction_root = _run_semantic_segmentation(
        image_paths=image_paths,
        semantic_class_maps=semantic_class_maps,
        exp_folder=args.experiment_folder,
        predictor=predictor,
        is_multiclass=args.dataset in MULTICLASS_SEMANTIC,
    )

    run_evaluation_for_semantic_segmentation(
        gt_paths=gt_paths,
        prediction_root=prediction_root,
        experiment_folder=args.experiment_folder,
        semantic_class_map=semantic_class_maps,
        is_multiclass=args.dataset in MULTICLASS_SEMANTIC,
    )

    _clear_files(experiment_folder=args.experiment_folder, semantic_class_maps=semantic_class_maps)


if __name__ == "__main__":
    main()
