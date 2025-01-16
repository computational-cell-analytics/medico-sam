# Finetuning Segment Anything for Medical Images

Scripts for training MedicoSAM (under the following directory structure):

- `generalists`: Training scripts for our generalist models.
    - `medsam`: Training scripts for MedSAM-style training.
    - `simplesam`: Training scripts for a simple training objective: using one point or box without iterative rectification in training.
- `specialists`: Training a specialist model on BTCV dataset.
