import os

from torch_em.data import datasets
from torch_em.data import MinInstanceSampler
from torch_em.transform.label import connected_components

from micro_sam.evaluation.model_comparison import generate_data_for_model_comparison, model_comparison


ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def compare_experiments_for_dataset(
    dataset_name,
    experiment_folder,
    standard_model,
    finetuned_model,
    intermediate_model=None,
    checkpoint1=None,
    checkpoint2=None,
    checkpoint3=None,
    view_napari=False,
    n_samples=20,
):
    output_folder = os.path.join(
        experiment_folder, "model_comparison", dataset_name, f"{standard_model}-{finetuned_model}"
    )
    plot_folder = os.path.join(experiment_folder, "candidates", dataset_name)
    if not os.path.exists(output_folder):
        loader = fetch_data_loaders(dataset_name)
        generate_data_for_model_comparison(
            loader=loader,
            output_folder=output_folder,
            model_type1=standard_model,
            model_type2=finetuned_model[:5],
            model_type3=intermediate_model[:5],
            n_samples=n_samples,
            checkpoint1=checkpoint1,
            checkpoint2=checkpoint2,
            checkpoint3=checkpoint3,
        )

    model_comparison(
        output_folder=output_folder,
        n_images_per_sample=10,
        min_size=100,
        plot_folder=plot_folder,
        point_radius=3,
        outline_dilation=0,
        have_model3=intermediate_model is not None
    )
    if view_napari:
        from micro_sam.evaluation.model_comparison import model_comparison_with_napari
        model_comparison_with_napari(output_folder, show_points=True)


def fetch_data_loaders(dataset_name):
    sampler = MinInstanceSampler()

    def _trafo(x, y):
        return x, y

    def _raw_trafo(raw):
        raw = raw.transpose(2, 0, 1)
        return raw

    def _label_trafo(labels):
        labels = labels.transpose(2, 0, 1)
        return labels

    if dataset_name == "camus":  # echocardiography
        loader = datasets.get_camus_loader(
            path=os.path.join(ROOT, "camus"), batch_size=1, patch_shape=(1, 512, 512), ndim=2, resize_inputs=True,
            sampler=sampler, shuffle=True, label_transform=connected_components, transform=_trafo,
        )

    elif dataset_name == "uwaterloo_skin":  # dermoscopy
        loader = datasets.get_uwaterloo_skin_loader(
            path=os.path.join(ROOT, "uwaterloo_skin"), batch_size=1, patch_shape=(512, 512), resize_inputs=True,
            sampler=sampler, shuffle=True, label_transform=connected_components, transform=_trafo,
        )

    elif dataset_name == "montgomery":  # x-ray
        loader = datasets.get_montgomery_loader(
            path=os.path.join(ROOT, "montgomery"), batch_size=1, patch_shape=(512, 512), resize_inputs=True,
            sampler=sampler, shuffle=True, label_transform=connected_components, transform=_trafo,
        )

    elif dataset_name == "piccolo":  # nbi
        loader = datasets.get_piccolo_loader(
            path=os.path.join(ROOT, "piccolo"), batch_size=1, patch_shape=(512, 512), split="test", resize_inputs=True,
            sampler=sampler, shuffle=True, label_transform=connected_components, transform=_trafo,
        )

    elif dataset_name == "cbis_ddsm":  # mammography
        loader = datasets.get_cbis_ddsm_loader(
            path=os.path.join(ROOT, "cbis_ddsm"), batch_size=1, patch_shape=(512, 512), resize_inputs=True, task="Mass",
            split="Test", tumour_type="MALIGNANT", shuffle=True, label_transform=connected_components, transform=_trafo,
        )

    elif dataset_name == "duke_liver":  # mri
        loader = datasets.get_duke_liver_loader(
            path=os.path.join(ROOT, "duke_liver"), batch_size=1, patch_shape=(1, 512, 512), ndim=2, resize_inputs=True,
            split="test", transform=_trafo, shuffle=True, sampler=sampler,
            raw_transform=_raw_trafo, label_transform=_label_trafo,
        )

    elif dataset_name == "papila":  # fundus
        loader = datasets.get_papila_loader(
            path=os.path.join(ROOT, "papila"), batch_size=1, patch_shape=(512, 512), resize_inputs=True, split="test",
            task="disc", sampler=sampler, shuffle=True,
        )

    elif dataset_name == "han-seg":  # ct
        loader = datasets.get_han_seg_loader(
            path=os.path.join(ROOT, "han-seg"), batch_size=1, patch_shape=(1, 512, 512), ndim=2, resize_inputs=True,
            sampler=sampler, label_transform=connected_components,
        )

    elif dataset_name == "microusp":  # micro-us
        loader = datasets.get_micro_usp_loader(
            path=os.path.join(ROOT, "microusp"), batch_size=1, patch_shape=(1, 512, 512), ndim=2,
            split="test", resize_inputs=True, sampler=sampler, shuffle=True,
        )

    else:
        raise ValueError(f"'{dataset_name}' is not a supported dataset.")

    return loader


def main(args):
    compare_experiments_for_dataset(
        dataset_name=args.dataset,
        experiment_folder="./figures",
        standard_model="vit_b",
        finetuned_model="vit_b_medicosam",
        intermediate_model="vit_b_medsam",
        checkpoint1=None,
        checkpoint2="/mnt/vast-nhr/projects/cidas/cca/models/medico-sam/multi_gpu/checkpoints/vit_b/medical_generalist_sam_multi_gpu/best_exported.pt",  # noqa
        checkpoint3="/mnt/vast-nhr/projects/cidas/cca/models/medsam/original/medsam_vit_b.pth",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    args = parser.parse_args()
    main(args)
