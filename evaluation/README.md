# Segment Anything Evaluation

Scripts for evaluating Segment Anything models and the finetuned `medico_sam` models.

Experimental plan:
- generalist evaluation for: (iterative prompting)
    - SAM +
    - medico-sam +
    - MedSAM (pretrained) +
    - SAM-Med2d (pretrained)
        - FT-SAM for sure +
        - adapter +
    - MedSAM (finetuned-our) +
        - 1 box only per object
    - SimpleSam (finetuned-our) +
        - Maceij's paper (randomly sample either a box or a positive point per object)


- semantic segmentation experiments (specific "limited" tasks)
    - any is fine (2d datasets and nnUNet-2d for this)
    - SAMed style segmentation +
            1. no prompts at all
            2. lora and full finetuning

        - different approaches
            - learnable prompt (comparable important backbones)
            - input image to prompt encoder (optional)
    - 3d segmentation
    - nnUNetv2


- Tool usage
    - napari tools (DKFZ, micro-sam)
    - 3d slicer (maceij's extension)
    - monai label (optional)


- Ablation:
    - single mask vs no mask
    - input size (optional)
