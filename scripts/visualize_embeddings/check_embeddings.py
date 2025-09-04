import numpy as np

from tukra.io import read_image

from torch_em.transform.raw import normalize
from torch_em.transform.generic import ResizeLongestSideInputs, ResizeInputs

from micro_sam.util import get_sam_model, precompute_image_embeddings
from micro_sam.visualization import project_embeddings_for_visualization


def check_embeddings(raw, model_type="vit_b", embedding_path=None, checkpoint=None, view=False):

    # Resize the images
    transform = ResizeLongestSideInputs(target_shape=(1024, 1024), is_rgb=(raw.ndim == 3))

    if raw.ndim == 2:
        raw = transform(raw)
    else:
        raw = transform(raw.transpose(2, 0, 1)).transpose(1, 2, 0)

    # Get the SAM predictor.
    predictor = get_sam_model(model_type=model_type, checkpoint_path=checkpoint)

    # Compute image embeddings.
    embeddings = precompute_image_embeddings(
        predictor=predictor,
        input_=raw,
        save_path=embedding_path,
        ndim=2,
    )

    # And create embedding visualizations.
    embedding_vis, _ = project_embeddings_for_visualization(
        image_embeddings=embeddings,
        n_components=3,
        as_rgb=False,
    )

    if view:
        import napari
        v = napari.Viewer()
        v.add_image(embedding_vis)
        napari.run()

    # Bring it up to (1024, 1024) first.
    etrafo = ResizeInputs(target_shape=(1024, 1024), is_rgb=True)
    embedding_vis = etrafo(embedding_vis.transpose(2, 0, 1)).transpose(1, 2, 0)

    if raw.ndim == 2:
        embedding_vis = transform.convert_transformed_inputs_to_original_shape(embedding_vis)
    else:
        embedding_vis = transform.convert_transformed_inputs_to_original_shape(
            embedding_vis.transpose(2, 0, 1)
        ).transpose(1, 2, 0)

    return embedding_vis


def main():
    # Load the image

    # 1. Brain MRI (PedIMS)
    # efname = "mri"
    # image_path = "/home/anwai/data/pedims/PediMS/P1/T1/processed/54714428_brain_FLAIR.nii.gz"
    # image = read_image(image_path)
    # image = normalize(image) * 255
    # image = image.transpose(2, 0, 1)[19]

    # 2. Abdomen CT (CURVAS)
    # efname = "ct"
    # image_path = "/media/anwai/ANWAI/data/curvas/training_set/UKCHLL001/image.nii.gz"
    # image = read_image(image_path)
    # image = normalize(image) * 255
    # image = image.transpose(2, 1, 0)[480]

    # 3. Narrow Band Imaging (PICCOLO)
    # efname = "nbi"
    # image_path = "/home/anwai/data/embedding_data/003_VP3_frame0051.png"
    # image = read_image(image_path)

    # 4. Dermoscopy (UWaterloo Skin)
    # efname = "dm"
    # image_path = "/home/anwai/data/embedding_data/SSM10_orig.jpg"
    # image = read_image(image_path)

    # 5. X-Ray (Montgomery)
    # efname = "xray"
    # image_path = "/home/anwai/data/embedding_data/MCUCXR_0001_0.png"
    # image = read_image(image_path)

    # 6. Mammography (CBIS DDSM)
    # efname = "mm"
    # image_path = "/home/anwai/data/embedding_data/Mass-Test_P_00194_RIGHT_CC_FULL_PRE.png"
    # image = read_image(image_path)

    # 7.Fundus (Papila)
    efname = "fd"
    image_path = "/home/anwai/data/embedding_data/RET010OD.jpg"
    image = read_image(image_path)

    # 8. Spine MRI (SPIDER)
    # efname = "sm"
    # image_path = "/home/anwai/data/embedding_data/104_t2.mha"
    # image = read_image(image_path).transpose(2, 0, 1)[6]
    # image = np.clip(image, 0, image.max())
    # image = normalize(image) * 255

    # 9. Ultrasound (CAMUS)
    # efname = "us"
    # image_path = "/home/anwai/data/embedding_data/camus_patient0012_2CH_half_sequence.nii_00015.tif"
    # image = read_image(image_path)[:, :, 0]
    # image = np.rot90(image, k=3)

    # View image
    view = False
    if view:
        import napari
        v = napari.Viewer()
        v.add_image(image)
        napari.run()

    embed_default = check_embeddings(raw=image, model_type="vit_b", embedding_path=f"{efname}.zarr")
    embed_medsam = check_embeddings(
        raw=image,
        model_type="vit_b",
        embedding_path=f"{efname}_medsam.zarr",
        checkpoint="/home/anwai/data/medsam_vit_b.pth",
    )
    embed_mi = check_embeddings(
        raw=image,
        model_type="vit_b",
        embedding_path=f"{efname}_mi.zarr",
        checkpoint="/media/anwai/ANWAI/models/medico_sam/model.pt",
    )

    # Change to 8bit.
    embed_default = (normalize(embed_default) * 255).astype("uint8")
    embed_medsam = (normalize(embed_medsam) * 255).astype("uint8")
    embed_mi = (normalize(embed_mi) * 255).astype("uint8")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(30, 20))

    kwargs = {}
    if efname == "ct":
        kwargs["vmin"] = 0.8 * image.max()
        kwargs["vmax"] = image.max()
    elif efname == "fd":
        image = (normalize(image) * 255).astype("uint8")

    ax[0].imshow(image, cmap="gray", **kwargs)
    ax[1].imshow(embed_default)
    ax[2].imshow(embed_medsam)
    ax[3].imshow(embed_mi)

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.savefig(f"./{efname}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"./{efname}.svg", dpi=600, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
