import torch

from medico_sam.models.monai_models import get_monai_models


def check_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy inputs
    image_2d = torch.zeros((1, 1, 1024, 1024), dtype=torch.float32).to(device)
    image_3d = torch.zeros((1, 1, 32, 512, 512), dtype=torch.float32).to(device)

    # First, let's get the 2d model.
    model = get_monai_models(
        image_size=(1024, 1024), in_channels=1, out_channels=3, ndim=2,
    )
    model.to(device)
    outputs = model(image_2d)
    print(outputs.shape)

    # Next, let's get the 3d model.
    model = get_monai_models(
        image_size=(32, 512, 512), in_channels=1, out_channels=3, ndim=3,
    )
    model.to(device)
    outputs = model(image_3d)
    print(outputs.shape)


def main():
    check_models()


if __name__ == "__main__":
    main()
