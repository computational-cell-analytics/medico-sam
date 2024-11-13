from torch_em.data.datasets import medical


DATA_ROOT = "/scratch/share/cidas/cca/data"
NNUNET_ROOT = "/scratch/share/cidas/cca/nnUNetv2"


def _get_paths(path, dataset, split):
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"'{split}' is not a valid split.")

    # 2d datasets
    if dataset == "oimhs":
        image_paths, gt_paths = medical.oimhs.get_oimhs_paths(path=path, split=split)

    elif dataset == "isic":
        image_paths, gt_paths = medical.isic.get_isic_paths(path=path, split=split)

    elif dataset == "dca1":
        image_paths, gt_paths = medical.dca1.get_dca1_paths(path=path, split=split)

    elif dataset == "cbis_ddsm":
        image_paths, gt_paths = medical.cbis_ddsm.get_cbis_ddsm_paths(path=path, split=split.title(), task="Mass")

    elif dataset == "drive":
        image_paths, gt_paths = medical.drive.get_drive_paths(path=path, split=split)

    elif dataset == "piccolo":
        image_paths, gt_paths = medical.piccolo.get_piccolo_paths(path, split="validation" if split == "val" else split)

    elif dataset == "siim_acr":
        image_paths, gt_paths = medical.siim_acr.get_siim_acr_paths(path=path, split=split)

    elif dataset == "hil_toothseg":
        image_paths, gt_paths = medical.hil_toothseg.get_hil_toothseg_paths(path=path, split=split)

    elif dataset == "covid_qu_ex":
        image_paths, gt_paths = medical.covid_qu_ex.get_covid_qu_ex_paths(path=path, split=split, task="lung")

    # 3d datasets
    elif dataset == "curvas":
        image_paths, gt_paths = medical.curvas.get_curvas_paths(path=path, split=split)

    elif dataset == "osic_pulmofib":
        image_paths, gt_paths = medical.osic_pulmofib.get_osic_pulmofib_paths(path=path, split=split)

    elif dataset == "sega":
        if split == "train":
            dchoice = "Rider"
        elif split == "val":
            dchoice = "Dongyang"
        elif split == "test":
            dchoice = "KiTS"

        image_paths, gt_paths = medical.sega.get_sega_paths(path=path, data_choice=dchoice)

    elif dataset == "duke_liver":
        image_paths, gt_paths = medical.duke_liver.get_duke_liver_paths(path=path, split=split)

    elif dataset == "toothfairy":
        image_paths, gt_paths = medical.toothfairy.get_toothfairy_paths(path=path, split=split, version="v1")

    elif dataset == "oasis":
        image_paths, gt_paths = medical.oasis.get_oasis_paths(path=path, split=split)

    elif dataset == "lgg_mri":
        image_paths, gt_paths = medical.lgg_mri.get_lgg_mri_paths(path=path, split=split)

    elif dataset == "leg_3d_us":
        image_paths, gt_paths = medical.leg_3d_us.get_leg_3d_us_paths(path=path, split=split)

    elif dataset == "micro_usp":
        image_paths, gt_paths = medical.micro_usp.get_micro_usp_paths(path=path, split=split)

    else:
        raise ValueError(dataset)

    return image_paths, gt_paths


def _binarise_labels(labels):
    labels = (labels > 1)
    labels = labels.astype("uint8")
    return labels


def _get_per_dataset_items(dataset):
    preprocess_inputs, preprocess_labels = None, None
    dataset_json_template = {
        "name": nnunet_dataset_name,
        "numTraining": len(val_ids) + len(train_ids),
    }

    # 2d dataset
    if dataset == "oimhs":
        file_suffix, transfer_mode = ".tif", "copy"

        dataset_json_template["channel_names"] = {"0": "R", "1": "G", "2": "B"}
        dataset_json_template["labels"] = {
            "background": 0,
            "choroid": 1,
            "retina": 2,
            "intraretinal_cysts": 3,
            "macular_hole": 4,
        }
        dataset_json_template["description"] = "OIMHS: https://doi.org/10.1038/s41597-023-02675-1"

    elif dataset == "isic":
        file_suffix, transfer_mode = ".tif", "store"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "R", "1": "G", "2": "B"}
        dataset_json_template["labels"] = {"background": 0, "lesion": 1}
        dataset_json_template["description"] = "ISIC: https://challenge.isic-archive.com/data/#2018"

    elif dataset == "dca1":
        file_suffix, transfer_mode = ".tif", "store"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "X-Ray Angiography"}
        dataset_json_template["labels"] = {"background": 0, "vessels": 1},
        dataset_json_template["description"] = "DCA1: https://doi.org/10.1038/s41597-023-02675-1"

    elif dataset == "cbis_ddsm":
        file_suffix, transfer_mode = ".tif", "store"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "Mammography"}
        dataset_json_template["labels"] = {"background": 0, "mass": 1},
        dataset_json_template["description"] = "CBIS-DDSM: https://doi.org/10.1038/sdata.2017.177"

    elif dataset == "drive":
        file_suffix, transfer_mode = ".tif", "store"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "R", "1": "G", "2": "B"}
        dataset_json_template["labels"] = {"background": 0, "vessels": 1},
        dataset_json_template["description"] = "DRIVE: https://drive.grand-challenge.org/"

    elif dataset == "piccolo":
        file_suffix, transfer_mode = ".tif", "store"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "R", "1": "G", "2": "B"}
        dataset_json_template["labels"] = {"background": 0, "lesion": 1},
        dataset_json_template["description"] = "PICCOLO: https://www.biobancovasco.bioef.eus/en/Sample-and-data-e-catalog/Databases/PD178-PICCOLO-EN1.html"  # noqa

    elif dataset == "siim_acr":
        ...

    elif dataset == "hil_toothseg":
        ...

    elif dataset == "covid_qu_ex":
        ...

    # 3d dataset
    elif dataset == "curvas":
        ...

    elif dataset == "osic_pulmofib":
        file_suffix, transfer_mode = ".nii.gz", "copy"

        dataset_json_template["channel_names"] = {"0": "CT"}
        dataset_json_template["labels"] = {"background": 0, "heart": 1, "lung": 2, "trachea": 3}
        dataset_json_template["description"] = "OSIC PulmoFib: https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data"  # noqa

    elif dataset == "sega":
        file_suffix, transfer_mode = ".nii.gz", "copy"

        dataset_json_template["channel_names"] = {"0": "CT"}
        dataset_json_template["labels"] = {"background": 0, "aorta": 1}
        dataset_json_template["description"] = "SegA: https://multicenteraorta.grand-challenge.org/"

    elif dataset == "duke_liver":
        file_suffix, transfer_mode = ".nii.gz", "copy"

        dataset_json_template["channel_names"] = {"0": "MRI"}
        dataset_json_template["labels"] = {"background": 0, "liver": 1}
        dataset_json_template["description"] = "Duke Liver: https://zenodo.org/records/7774566"

    elif dataset == "toothfairy":
        ...

    elif dataset == "oasis":
        ...

    elif dataset == "lgg_mri":
        ...

    elif dataset == "leg_3d_us":
        ...

    elif dataset == "micro_usp":
        ...

    else:
        raise ValueError(dataset)

    dataset_json_template["file_ending"] = file_suffix

    return file_suffix, transfer_mode, dataset_json_template, preprocess_inputs, preprocess_labels
