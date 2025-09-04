# Experimental Scripts

<b> Important!!!!!! Read this before using this code or creating an issue. </b>

This folder contains finetuning and evaluation code for applying Segment Anything model to medical imaging data using the `medico_sam` and `micro_sam` libraries. This code was used for our experiments in the publication, but it may become outdated due to changes in function signatures, etc., and often does not use the functionality that we recommend to users. We also don't actively maintain the code here. Please refer to the [example scripts](https://github.com/computational-cell-analytics/medico-sam/tree/master/examples) for well maintaned and documented `medico_sam` examples.

<b> Important!!!!!! Read this before using this code or creating an issue. </b>

## Finetuning and Evaluation Scripts

The subfolders contain the code for different finetuning and evaluation experiments for microscopy data:
- `evaluation`: Contains scripts for evaluating both interactive segmentation (2d and 3d) and semantic segmentation (2d and 3d).
- `finetuning`: Contains scripts for finetuning SAM for medical images with several methods.
- `semantic_segmentation`: Contains scripts for training and evaluating semantic segmentation using SAM-based methods and other benchmark methods.
