import os
from collections import OrderedDict

import torch

from micro_sam import util
import torch.version


def export_sam_models(checkpoint_path):
    prefix = "module.sam."  # DDP prefix to be filtered out.

    # 1. Load model state.
    # 2. Convert weight keys to match the sam expectation.
    state, model_state = util._load_checkpoint(checkpoint_path=checkpoint_path)
    decoder_state = state["decoder_state"]

    # Make them dictionaries.
    model_state = OrderedDict([(k[len(prefix):] if k.startswith(prefix) else k, v) for k, v in model_state.items()])
    decoder_state = OrderedDict([(k[len(prefix):] if k.startswith(prefix) else k, v) for k, v in decoder_state.items()])

    # And now save all stuff
    torch.save(
        {"model_state": model_state, "decoder_state": decoder_state},
        os.path.join(os.path.dirname(checkpoint_path), "model.pt")
    )


def main():
    # Full data model.
    export_sam_models(
        "/mnt/vast-nhr/projects/cidas/cca/models/medico-sam/v2/multi_gpu/checkpoints/vit_b/medical_generalist_sam_multi_gpu/best.pt"  # noqa
    )

    # 50% data model.
    export_sam_models(
        "/mnt/vast-nhr/projects/cidas/cca/models/medico-sam/v2/multi_gpu/checkpoints/vit_b/medical_generalist_sam_multi_gpu_0.5/best.pt"  # noqa
    )


if __name__ == "__main__":
    main()
