from torch_em.data.datasets import medical


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
NNUNET_ROOT = "/mnt/vast-nhr/projects/cidas/cca/nnUNetv2"


def _get_paths(path, dataset, split):
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"'{split}' is not a valid split.")

    # NOTE: Case for 'sega' dataset where the splits are on data-level.
    if dataset == "sega":
        if split == "train":
            dchoice = "Rider"
        elif split == "val":
            dchoice = "Dongyang"
        elif split == "test":
            dchoice = "KiTS"
        else:
            raise ValueError

    dpaths = {
        # 2d datasets
        "oimhs": lambda: medical.oimhs.get_oimhs_paths(path=path, split=split),
        "isic": lambda: medical.isic.get_isic_paths(path=path, split=split),
        "dca1": lambda: medical.dca1.get_dca1_paths(path=path, split=split),
        "cbis_ddsm": lambda: medical.cbis_ddsm.get_cbis_ddsm_paths(
            path=path, split=split.title(), task="Mass", ignore_mismatching_pairs=True,
        ),
        "drive": lambda: medical.drive.get_drive_paths(path=path, split=split),
        "piccolo": lambda: medical.piccolo.get_piccolo_paths(path, split="validation" if split == "val" else split),
        "siim_acr": lambda: medical.siim_acr.get_siim_acr_paths(path=path, split=split),
        "hil_toothseg": lambda: medical.hil_toothseg.get_hil_toothseg_paths(path=path, split=split),
        "covid_qu_ex": lambda: medical.covid_qu_ex.get_covid_qu_ex_paths(path=path, split=split, task="lung"),
        # 3d datasets
        "curvas": lambda: medical.curvas.get_curvas_paths(path=path, split=split),
        "osic_pulmofib": lambda: medical.osic_pulmofib.get_osic_pulmofib_paths(path=path, split=split),
        "sega": lambda: medical.sega.get_sega_paths(path=path, data_choice=dchoice),
        "duke_liver": lambda: medical.duke_liver.get_duke_liver_paths(path=path, split=split),
        "toothfairy": lambda: medical.toothfairy.get_toothfairy_paths(path=path, split=split, version="v1"),
        "oasis": lambda: medical.oasis.get_oasis_paths(path=path, split=split),
        "lgg_mri": lambda: medical.lgg_mri.get_lgg_mri_paths(path=path, split=split),
        "leg_3d_us": lambda: medical.leg_3d_us.get_leg_3d_us_paths(path=path, split=split),
        "micro_usp": lambda: medical.micro_usp.get_micro_usp_paths(path=path, split=split),
    }

    if dataset not in dpaths:
        raise ValueError(f"'{dataset}' is not a supported dataset.")

    input_paths = dpaths[dataset]()
    if dataset == "lgg_mri":
        image_paths = gt_paths = input_paths
    else:
        image_paths, gt_paths = input_paths

    return image_paths, gt_paths


def _binarise_labels(labels):
    labels = (labels > 1)
    labels = labels.astype("uint8")
    return labels


def _get_per_dataset_items(dataset, nnunet_dataset_name, train_id_count, val_id_count):
    preprocess_inputs, preprocess_labels = None, None  # Decides via a callable whether to perform some preprocessing.
    keys = None  # Decide for container data structures for tukra's image reader.
    dataset_json_template = {
        "name": nnunet_dataset_name,
        "numTraining": val_id_count + train_id_count,
    }

    # 2d dataset
    if dataset == "oimhs":
        file_suffix, transfer_mode = ".tif", "copy"

        dataset_json_template["channel_names"] = {"0": "R", "1": "G", "2": "B"}
        dataset_json_template["labels"] = {
            "background": 0, "choroid": 1, "retina": 2, "intraretinal_cysts": 3, "macular_hole": 4
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
        dataset_json_template["labels"] = {"background": 0, "vessels": 1}
        dataset_json_template["description"] = "DCA1: https://doi.org/10.1038/s41597-023-02675-1"

    elif dataset == "cbis_ddsm":
        file_suffix, transfer_mode = ".tif", "store"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "Mammography"}
        dataset_json_template["labels"] = {"background": 0, "mass": 1}
        dataset_json_template["description"] = "CBIS-DDSM: https://doi.org/10.1038/sdata.2017.177"

    elif dataset == "drive":
        file_suffix, transfer_mode = ".tif", "store"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "R", "1": "G", "2": "B"}
        dataset_json_template["labels"] = {"background": 0, "vessels": 1}
        dataset_json_template["description"] = "DRIVE: https://drive.grand-challenge.org/"

    elif dataset == "piccolo":
        file_suffix, transfer_mode = ".tif", "store"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "R", "1": "G", "2": "B"}
        dataset_json_template["labels"] = {"background": 0, "lesion": 1}
        dataset_json_template["description"] = "PICCOLO: https://www.biobancovasco.bioef.eus/en/Sample-and-data-e-catalog/Databases/PD178-PICCOLO-EN1.html"  # noqa

    elif dataset == "siim_acr":
        file_suffix, transfer_mode = ".tif", "store"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "Chest X-Ray"}
        dataset_json_template["labels"] = {"background": 0, "pneumothorax": 1}
        dataset_json_template["description"] = "SIIM ACR: https://kaggle.com/competitions/siim-acr-pneumothorax-segmentation."  # noqa

    elif dataset == "hil_toothseg":
        file_suffix, transfer_mode = ".tif", "store"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "Panoramic Radiographs"}
        dataset_json_template["labels"] = {"background": 0, "teeth": 1}
        dataset_json_template["description"] = "HIL ToothSeg: https://www.mdpi.com/1424-8220/21/9/3110."

    elif dataset == "covid_qu_ex":
        file_suffix, transfer_mode = ".tif", "store"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "Chest X-Ray"}
        dataset_json_template["labels"] = {"background": 0, "lung": 1}
        dataset_json_template["description"] = "Covid QU EX: https://doi.org/10.1016/j.compbiomed.2021.104319"

    # 3d dataset
    elif dataset == "curvas":
        file_suffix, transfer_mode = ".nii.gz", "copy"

        dataset_json_template["channel_names"] = {"0": "CT"}
        dataset_json_template["labels"] = {"background": 0, "pancreas": 1, "kidney": 2, "liver": 3}
        dataset_json_template["description"] = "CURVAS: https://curvas.grand-challenge.org"

    elif dataset == "osic_pulmofib":
        file_suffix, transfer_mode = ".nii.gz", "copy"

        dataset_json_template["channel_names"] = {"0": "CT"}
        dataset_json_template["labels"] = {"background": 0, "heart": 1, "lung": 2, "trachea": 3}
        dataset_json_template["description"] = "OSIC PulmoFib: https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data"  # noqa

    elif dataset == "sega":
        file_suffix, transfer_mode = ".nii.gz", "copy"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "CT"}
        dataset_json_template["labels"] = {"background": 0, "aorta": 1}
        dataset_json_template["description"] = "SegA: https://multicenteraorta.grand-challenge.org/"

    elif dataset == "duke_liver":
        file_suffix, transfer_mode = ".nii.gz", "copy"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "MRI"}
        dataset_json_template["labels"] = {"background": 0, "liver": 1}
        dataset_json_template["description"] = "Duke Liver: https://zenodo.org/records/7774566"

    elif dataset == "toothfairy":
        file_suffix, transfer_mode = ".nii.gz", "copy"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "CBCT"}
        dataset_json_template["labels"] = {"background": 0, "mandibular canal": 1}
        dataset_json_template["description"] = "ToothFairy v1: https://toothfairy.grand-challenge.org/"

    elif dataset == "oasis":
        file_suffix, transfer_mode = ".nii.gz", "copy"

        dataset_json_template["channel_names"] = {"0": "T1 Brain MRI"}
        dataset_json_template["labels"] = {"background": 0, "gray matter": 1, "thalamus": 2, "white matter": 3, "csf": 4}  # noqa
        dataset_json_template["description"] = "OASIS: https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md"  # noqa

    elif dataset == "lgg_mri":
        file_suffix, transfer_mode = ".nii.gz", "store"
        preprocess_labels = _binarise_labels
        keys = ("raw/flair", "labels")

        dataset_json_template["channel_names"] = {"0": "FLAIR MRI"}
        dataset_json_template["labels"] = {"background": 0, "low grade glioma": 1}
        dataset_json_template["description"] = "LGG MRI: https://www.nejm.org/doi/full/10.1056/NEJMoa1402121"

    elif dataset == "leg_3d_us":
        file_suffix, transfer_mode = ".nii.gz", "store"

        dataset_json_template["channel_names"] = {"0": "FLAIR MRI"}
        dataset_json_template["labels"] = {"background": 0, "SOL": 1, "GM": 2, "GL": 3}
        dataset_json_template["description"] = "LEG 3D US: https://doi.org/10.1371/journal.pone.0268550"

    elif dataset == "micro_usp":
        file_suffix, transfer_mode = ".nii.gz", "copy"
        preprocess_labels = _binarise_labels

        dataset_json_template["channel_names"] = {"0": "Micro US"}
        dataset_json_template["labels"] = {"background": 0, "prostate": 1}
        dataset_json_template["description"] = "MicroUSP: https://doi.org/10.1016/j.compmedimag.2024.102326"

    else:
        raise ValueError(dataset)

    dataset_json_template["file_ending"] = file_suffix

    return file_suffix, transfer_mode, dataset_json_template, preprocess_inputs, preprocess_labels, keys
