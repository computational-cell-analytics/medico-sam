import os

from medico_sam.evaluation import inference
# from medico_sam.evaluation.evaluation import run_evaluation_for_semantic_segmentation

from micro_sam.util import get_sam_model

from util import get_dataset_paths, get_default_arguments, _clear_files


def _run_semantic_segmentation(image_paths, semantic_class_maps, exp_folder, predictor):
    prediction_root = os.path.join(exp_folder, "semantic_segmentation")
    embedding_folder = None  # HACK: compute embeddings on-the-fly now, else: os.path.join(exp_folder, "embeddings")
    inference.run_semantic_segmentation(
        predictor=predictor,
        image_paths=image_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_root,
        semantic_class_map=semantic_class_maps,
    )
    return prediction_root


def main():
    args = get_default_arguments()

    # get the predictor to perform inference
    predictor = get_sam_model(model_type=args.model, checkpoint_path=args.checkpoint)

    image_paths, gt_paths, semantic_class_maps = get_dataset_paths(dataset_name=args.dataset, split="test")

    # HACK: testing it on first 200 (or fewer) samples
    image_paths, gt_paths = image_paths[:200], gt_paths[:200]

    prediction_root = _run_semantic_segmentation(
        image_paths=image_paths,
        semantic_class_maps=semantic_class_maps,
        exp_folder=args.experiment_folder,
        predictor=predictor,
    )

    # run_evaluation_for_semantic_segmentation(
    #     gt_paths=gt_paths,
    #     prediction_root=prediction_root,
    #     experiment_folder=args.experiment_folder,
    #     semantic_class_map=semantic_class_maps,
    # )

    _clear_files(experiment_folder=args.experiment_folder, semantic_class_maps=semantic_class_maps)


if __name__ == "__main__":
    main()
