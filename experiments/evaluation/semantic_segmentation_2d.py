import os
from glob import glob
from natsort import natsorted

from medico_sam.evaluation import inference
from medico_sam.util import get_medico_sam_model
from medico_sam.evaluation.evaluation import run_evaluation_for_semantic_segmentation

from util import get_default_arguments, _clear_files, MULTICLASS_SEMANTIC


CLASS_MAPS = {
    # 2d datasets
    "oimhs": {"choroid": 1, "retina": 2, "intraretinal_cysts": 3, "macular_hole": 4},
    "isic": {"skin_lesion": 1},
    "dca1": {"vessel": 1},
    "cbis_ddsm": {"mass": 1},
    "piccolo": {"polyps": 1},
    "hil_toothseg": {"teeth": 1},
}

DATASET_MAPPING_2D = {
    "oimhs": "Dataset201_OIMHS",
    "isic": "Dataset202_ISIC",
    "dca1": "Dataset203_DCA1",
    "cbis_ddsm": "Dataset204_CBISDDSM",
    "piccolo": "Dataset206_PICCOLO",
    "hil_toothseg": "Dataset208_HIL_ToothSeg",
}


def _run_semantic_segmentation(image_paths, semantic_class_maps, exp_folder, predictor, is_multiclass):
    prediction_root = os.path.join(exp_folder, "semantic_segmentation")
    embedding_folder = None  # NOTE: computes embeddings on-the-fly now
    inference.run_semantic_segmentation(
        predictor=predictor,
        image_paths=image_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_root,
        semantic_class_map=semantic_class_maps,
        is_multiclass=is_multiclass,
    )
    return prediction_root


def get_2d_dataset_paths(dataset_name):
    root_dir = os.path.join("/mnt/vast-nhr/projects/cidas/cca/nnUNetv2/nnUNet_raw", DATASET_MAPPING_2D[dataset_name])
    image_paths = natsorted(glob(os.path.join(root_dir, "imagesTs", "*")))
    gt_paths = natsorted(glob(os.path.join(root_dir, "labelsTs", "*")))
    assert len(image_paths) == len(gt_paths)
    return image_paths, gt_paths, CLASS_MAPS[dataset_name]


def main():
    args = get_default_arguments()

    image_paths, gt_paths, semantic_class_maps = get_2d_dataset_paths(dataset_name=args.dataset)

    # get the predictor to perform inference
    predictor = get_medico_sam_model(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        flexible_load_checkpoint=True,
        num_multimask_outputs=(len(semantic_class_maps.keys()) + 1),
        peft_kwargs={"rank": args.lora_rank} if args.lora_rank is not None else None,
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
        ensure_channels_first=False,
    )

    _clear_files(experiment_folder=args.experiment_folder, semantic_class_maps=semantic_class_maps)


if __name__ == "__main__":
    main()
