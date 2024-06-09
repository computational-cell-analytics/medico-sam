import os

from torch_em.data.datasets import medical

from preprocess_datasets import ROOT


VALID_DATASETS = [
    "sega", "uwaterloo_skin", "idrid", "camus",
]


def get_input_paths(dataset_name):
    image_paths = ...
    gt_paths = ...

    return image_paths, gt_paths
