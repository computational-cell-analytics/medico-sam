# Segment Anything Evaluation

Scripts for evaluating Segment Anything models and the finetuned `medico_sam` models.

Experimental plan:
- generalist evaluation for: (iterative prompting)
    - SAM
    - medico-sam
    - MedSAM (pretrained)
    - SAM-Med2d (pretrained)
        - FT-SAM for sure
        - (optional) adapter
    - MedSAM (finetuned-our)
        - 1 box only - BCE-DICE
    - SimpleSam (finetuned-our)
        - Maceij's paper (randomly sample either a box or a positive point) - (check the loss function)


- semantic segmentation experiments (specific "limited" tasks)
    - SAMed style segmentation
        - different approaches
            - learnable prompt (comparable important backbones)
            - input image to prompt encoder (optional)
    - 3d segmentation
    - nnUNetv2


- Tool usage
    - napari tools (DKFZ, micro-sam)
    - 3d slicer (maceij's extension)
    - monai label (optional)
