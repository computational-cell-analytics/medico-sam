# MedicoSAM: Towards foundation models for medical image segmentation

<a href="https://github.com/computational-cell-analytics/medico-sam"><img src="https://github.com/computational-cell-analytics/medico-sam/blob/master/docs/logos/logo.png" width="200" align="right">

MedicoSAM implements interactive annotation and (automatic) semantic segmentation for medical images. It is built on top of [Segment Anything](https://segment-anything.com/) by Meta AI and specializes it for biomedical imaging data. Its core components are:
- The `medico_sam` publicly available model for interactive data annotation in 2d and 3d data that are fine-tuned on publicly available medical images.
- The `medico_sam` library provides training frameworks, inspired by [Segment Anything for Microscopy](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html), for downstream tasks:
  - Apply Segment Anything to 2d and 3d data or fine-tune it on your data.
  - Supports semantic segmentation for 2d and 3d data.

Based on these components, `medico_sam` enables fast interactive and automatic annotation for medical images.

## Installation

How to install `medico-sam` python library from source?

To create the environment and install `medico_sam` into it follow these steps:

1. Clone the repository: `git clone https://github.com/computational-cell-analytics/medico-sam`
2. Enter it: `cd medico-sam`
3. Create the environment with the necessary requirements: `conda env create -f environment.yaml`
4. Activate the environment: `conda activate medico-sam`
5. Install `medico_sam`: `pip install -e .`

## Download Model Checkpoints

You can find the model checkpoints at: https://owncloud.gwdg.de/index.php/s/AB69HGhj8wuozXQ

Download it via terminal using: `wget https://owncloud.gwdg.de/index.php/s/AB69HGhj8wuozXQ/download -O vit_b_medicosam.pt`.

## Tool Usage for Interactive Annotation

See [`TOOL_USAGE.md`](./TOOL_USAGE.md) document for details.

> TLDR: We recommend using our model with [`micro-sam`](https://github.com/computational-cell-analytics/micro-sam) annotator tool, in terms of compatibility and ease of annotation experience!

## Citation
If you are using this repository in your research please cite:

- our [preprint](https://doi.org/10.48550/arXiv.2501.11734).
- the [Segment Anything fo Microscopy](https://www.nature.com/articles/s41592-024-02580-4) publication.
- and the original [Segment Anything](https://arxiv.org/abs/2304.02643) publication.
