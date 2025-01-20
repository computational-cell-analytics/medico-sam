import os


if os.path.exists("/scratch/usr/nimanwai"):
    MODELS_DIR = "/scratch/usr/nimanwai/models/sam2"
else:
    MODELS_DIR = "/media/anwai/ANWAI/models/sam2"


CHECKPOINT_PATHS = {
    "sam2.0": {
        "hvit_t": os.path.join(MODELS_DIR, "sam2_hiera_tiny.pt"),
        "hvit_s": os.path.join(MODELS_DIR, "sam2_hiera_small.pt"),
        "hvit_b": os.path.join(MODELS_DIR, "sam2_hiera_base_plus.pt"),
        "hvit_l": os.path.join(MODELS_DIR, "sam2_hiera_large.pt"),
    },
    "sam2.1": {
        "hvit_t": os.path.join(MODELS_DIR, "sam2.1_hiera_tiny.pt"),
        "hvit_s": os.path.join(MODELS_DIR, "sam2.1_hiera_small.pt"),
        "hvit_b": os.path.join(MODELS_DIR, "sam2.1_hiera_base_plus.pt"),
        "hvit_l": os.path.join(MODELS_DIR, "sam2.1_hiera_large.pt"),
    }
}
