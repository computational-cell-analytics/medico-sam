import os

from medico_sam.evaluation import inference
from medico_sam.evaluation.evaluation import run_evaluation_for_iterative_prompting_per_semantic_class

from micro_sam.util import get_sam_model

from util import get_dataset_paths, get_default_arguments, _clear_files


def _run_iterative_prompting(
    image_paths, gt_paths, semantic_class_maps, exp_folder, predictor, start_with_box_prompt, use_masks
):
    prediction_root = os.path.join(
        exp_folder, "start_with_box" if start_with_box_prompt else "start_with_point"
    )
    embedding_folder = os.path.join(exp_folder, "embeddings")
    inference.run_inference_with_iterative_prompting_per_semantic_class(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=gt_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_root,
        start_with_box_prompt=start_with_box_prompt,
        use_masks=use_masks,
        semantic_class_map=semantic_class_maps,
    )
    return prediction_root


def _evaluate_iterative_prompting(
    gt_paths, semantic_class_maps, prediction_root, start_with_box_prompt, exp_folder
):
    run_evaluation_for_iterative_prompting_per_semantic_class(
        gt_paths=gt_paths,
        prediction_root=prediction_root,
        experiment_folder=exp_folder,
        start_with_box_prompt=start_with_box_prompt,
        semantic_class_map=semantic_class_maps,
    )


def main():
    args = get_default_arguments()

    start_with_box_prompt = args.box  # overwrite to start first iters' prompt with box instead of single point

    # get the predictor to perform inference
    predictor = get_sam_model(model_type=args.model, checkpoint_path=args.checkpoint)

    image_paths, gt_paths, semantic_class_maps = get_dataset_paths(dataset_name=args.dataset, split="test")

    prediction_root = _run_iterative_prompting(
        image_paths=image_paths,
        gt_paths=gt_paths,
        semantic_class_maps=semantic_class_maps,
        exp_folder=args.experiment_folder,
        predictor=predictor,
        start_with_box_prompt=start_with_box_prompt,
        use_masks=args.use_masks
    )

    _evaluate_iterative_prompting(
        gt_paths=gt_paths,
        semantic_class_maps=semantic_class_maps,
        prediction_root=prediction_root,
        start_with_box_prompt=start_with_box_prompt,
        exp_folder=args.experiment_folder,
    )

    _clear_files(experiment_folder=args.experiment_folder, semantic_class_maps=semantic_class_maps)


if __name__ == "__main__":
    main()
