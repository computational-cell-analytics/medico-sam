import os
from collections import OrderedDict

import torch

from micro_sam.util import _load_checkpoint, get_sam_model
from micro_sam.instance_segmentation import get_unetr


def run_inference():
    export_path = os.path.join("checkpoints", "vit_b", "livecell_sam_multi_gpu", "model.pt")
    if not os.path.exists(export_path):
        # Export model in a specific manner for both SAM and UNETR decoder.
        state, model_state = _load_checkpoint(
            checkpoint_path=os.path.join("checkpoints", "vit_b", "livecell_sam_multi_gpu", "best.pt")
        )
        model_prefix = "module.sam."
        model_state = OrderedDict(
            [(k[len(model_prefix):] if k.startswith(model_prefix) else k, v) for k, v in model_state.items()]
        )

        decoder_prefix = "module."
        decoder_state = state["decoder_state"]
        decoder_state = OrderedDict(
            [
                (k[len(decoder_prefix):] if k.startswith(decoder_prefix) else k, v)
                for k, v in decoder_state.items() if "encoder" not in k
            ]
        )

        save_state = {"model_state": model_state, "decoder_state": decoder_state}
        torch.save(save_state, export_path)

    # Now, let's validate if we can load the 'export_path' model into the backbone, ready for evaluation.
    predictor, state = get_sam_model(checkpoint_path=export_path, return_state=True)
    print("Can load SAM model.")

    get_unetr(
        image_encoder=predictor.model.image_encoder,
        decoder_state=state["decoder_state"],
    )
    print("Can load UNETR model.")


def main():
    run_inference()


if __name__ == "__main__":
    main()
