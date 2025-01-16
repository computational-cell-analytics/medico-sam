import os
import xxhash
import argparse
from collections import OrderedDict

from micro_sam.util import _load_checkpoint

import torch


BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

OUTPUT_FOLDER = "/mnt/vast-nhr/projects/cidas/cca/models/medico-sam/exported_models"


def compute_checksum(path):
    xxh_checksum = xxhash.xxh128()
    with open(path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            xxh_checksum.update(data)

    return xxh_checksum.hexdigest()


def export_model(model_path, model_type):
    assert os.path.exists(model_path), "The filepath to the model checkpoint does not exist."

    model_name = f"{model_type}_medical_imaging"
    output_folder = os.path.join(OUTPUT_FOLDER, "medical_imaging", model_name)
    if os.path.exists(output_folder):
        print("The model", model_name, "has already been exported.")
        return

    os.makedirs(output_folder, exist_ok=True)

    exported_path = os.path.join(
        output_folder, f"{model_type}.pt"
    )

    _, model_state = _load_checkpoint(checkpoint_path=model_path)

    sam_prefix = "module.sam."
    model_state = OrderedDict(
        [(k[len(sam_prefix):] if k.startswith(sam_prefix) else k, v) for k, v in model_state.items()]
    )
    torch.save(model_state, exported_path)

    # Test loading the medico-sam model.
    from medico_sam.util import get_medico_sam_model
    _ = get_medico_sam_model(model_type=model_type, checkpoint_path=exported_path)

    print("Exported model", model_name)
    encoder_checksum = compute_checksum(exported_path)
    print("Encoder:")
    print(model_name, f"xxh128:{encoder_checksum}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True, type=str)
    parser.add_argument("-m", "--model_type", required=True, type=str)
    args = parser.parse_args()

    export_model(model_path=args.checkpoint, model_type=args.model_type)


if __name__ == "__main__":
    main()
