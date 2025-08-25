import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import numpy as np

from tukra.io import read_image
from tukra.inference import get_biomedparse

from elf.evaluation import dice_score


CLASS_MAPS = {
    # 2d datasets
    # NOTE for OIMHS: edema, also coined as custoid macular edema, are form of intraretinal cysts:
    # here's a reference paper for this: https://doi.org/10.1136/bjophthalmol-2014-305305
    "oimhs": {"edema": 1},  # NOTE: Only one out of four classes available for this data.
    "isic": {"lesion": 1},
    "piccolo": {"neoplastic polyp": 1, "non-neoplastic polyp": 1},  # NOTE: Done to map polyps to one class for eval.

    # "dca1": {"vessel": 1},
    # "cbis_ddsm": {"mass": 1},
    # "hil_toothseg": {"teeth": 1},
}

DATASET_MAPPING_2D = {
    "oimhs": "Dataset201_OIMHS",
    "isic": "Dataset202_ISIC",
    "piccolo": "Dataset206_PICCOLO",

    # "dca1": "Dataset203_DCA1",
    # "cbis_ddsm": "Dataset204_CBISDDSM",
    # "hil_toothseg": "Dataset208_HIL_ToothSeg",
}


def get_2d_dataset_paths(dataset_name):
    root_dir = os.path.join(
        "/mnt/vast-nhr/projects/cidas/cca/nnUNetv2/nnUNet_raw", DATASET_MAPPING_2D[dataset_name]
    )
    image_paths = natsorted(glob(os.path.join(root_dir, "imagesTs", "*")))
    gt_paths = natsorted(glob(os.path.join(root_dir, "labelsTs", "*")))

    assert len(image_paths) == len(gt_paths)
    return image_paths, gt_paths, CLASS_MAPS[dataset_name]


def run_biomedparse_prediction(image, gt, modality, class_maps):
    model = get_biomedparse.get_biomedparse_model()  # Get the biomedparse model.

    # Run inference per image.
    prediction = get_biomedparse.run_biomedparse_automatic_inference(
        input_path=image, modality_type=modality, model=model, verbose=False,
    )

    semantic_seg = np.zeros_like(gt, dtype="uint8")
    if prediction is not None:
        prompts = list(prediction.keys())  # Extracting detected classes.
        segmentations = list(prediction.values())  # Extracting the segmentations.

        # Map all predicted labels.
        for prompt, curr_seg in zip(prompts, segmentations):
            semantic_seg[curr_seg > 0] = class_maps[prompt]

    return semantic_seg


def main():
    # NOTE: Below are datasets for which BioMedParse could be run for (in an automatic fashion)!
    # dataset_name, modality = "piccolo", "Endoscopy"  # Mean score for polyps: 0.8459717204324456
    # dataset_name, modality = "isic", "Dermoscopy"  # Mean score for lesions: 
    dataset_name, modality = "oimhs", "OCT"  # Mean score for CME: 0.7079523903144189

    image_paths, gt_paths, class_maps = get_2d_dataset_paths(dataset_name)

    scores = []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = read_image(image_path)
        gt = read_image(gt_path)

        if dataset_name == "oimhs":
            gt = (gt == 3).astype("uint8")

        if len(np.unique(gt)) == 1:  # NOTE: For images with no foreground, there's no point running the model.
            continue

        prediction = run_biomedparse_prediction(image, gt, modality, class_maps)

        visualize = False
        if visualize:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(20, 10))
            ax[0].imshow(image)
            ax[1].imshow(gt)
            ax[2].imshow(prediction)
            plt.savefig("./test.png")

        score = dice_score(prediction, gt)
        scores.append(score)

    breakpoint()

    final_score = np.mean(scores)
    print(final_score)


if __name__ == "__main__":
    main()
