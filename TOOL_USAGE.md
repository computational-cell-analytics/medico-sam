## Tool Usage for Interactive Annotation

### 1. [`micro-sam`](https://github.com/computational-cell-analytics/micro-sam) (napari-based annotation tool):

> *Recommended Tool for Best Compatibility*

- When installing the `medico_sam` library, you will have access to `micro-sam` without any further installation.
- In terminal, open `napari`.
- Go to the top `Plugins` menu -> `Segment Anything for Microscopy` -> choice of annotator (`Annotator 2d` for annotating 2d images and `Annotator 3d` for annotating 3d images)
- Provide the filepath to downloaded model checkpoints to `Embedding Settings` drop-down -> `custom weights path` and start annotating your images.
- Visit the documentation for more details on annotation workflows.

> NOTE: We plan to integrate our model with `micro-sam` soon!

### 2. [`napari-sam`](https://github.com/MIC-DKFZ/napari-sam) (napari-based annotation tool):

> Expects some minor code changes to provide custom filepaths to finetuned models.

- Install `napari-sam` from source by following the installation instructions [here](https://github.com/MIC-DKFZ/napari-sam?tab=readme-ov-file#installation).
- Add the model filepaths and name for the model to the dictionary located at line: https://github.com/MIC-DKFZ/napari-sam/blob/main/src/napari_sam/_widget.py#L43-L49 (see below mentioned example detailing the updates)
- Visit the documentation for mroe details on annotation workflows.

The dictionary after changes should look like:
```python
 SAM_MODELS = {
     "default": {"filename": "sam_vit_h_4b8939.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "model": build_sam_vit_h},
     "vit_h": {"filename": "sam_vit_h_4b8939.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "model": build_sam_vit_h},
     "vit_l": {"filename": "sam_vit_l_0b3195.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "model": build_sam_vit_l},
     "vit_b": {"filename": "sam_vit_b_01ec64.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "model": build_sam_vit_b},
     "MedSAM": {"filename": "sam_vit_b_01ec64_medsam.pth", "url": "https://syncandshare.desy.de/index.php/s/yLfdFbpfEGSHJWY/download/medsam_20230423_vit_b_0.0.1.pth", "model": build_sam_vit_b},
     "MedicoSAM": {"filename": "<MODEL_NAME>.pt", "url": None, "model": build_sam_vit_b},  # NEW LINE
 }
```

### 3. [`samm`: Segment Any Medical-Model](https://github.com/bingogome/samm) (Slicer-based annotation tool):

> Expects some code changes to provide custom filepaths to finetuned models and a few minor adaptation to run on CPU resources.

- Install `samm` from source by following the installation instructions [here](https://github.com/bingogome/samm/tree/main?tab=readme-ov-file#installation-guide).
- Below listed are the code changes required while testing custom finetuned models:
  - Create a folder under `samm/samm-python-terminal` with the name: `mkdir samm-workspace`
  - Move the model checkpoints in `samm-workspace`
  - Update the filepath to `self.sam_checkpoint` at https://github.com/bingogome/samm/blob/main/samm-python-terminal/sam_server.py#L26.
  - Add a new model map at https://github.com/bingogome/samm/blob/main/samm-python-terminal/utl_sam_msg.py#L233-L240 and https://github.com/bingogome/samm/blob/main/samm/SammBase/SammBaseLib/UtilMsgFactory.py#L233-L240, such as it looks like:
  ```python
  SammModelMapper = {
    "vit_b" : 0,
    "vit_l" : 1,
    "vit_h" : 2,
    "mobile_vit_t" : 3,
    "medsam_vit_b" : 4,
    "medicosam_vit_b": 5,  # NEW LINE
    "DICT" : ["vit_b", "vit_l", "vit_h", "mobile_vit_t", "medsam_vit_b", "medicosam_vit_b"]  # NEW MODEL NAME ADDITION
  }
  ```
  - Add the model choice at https://github.com/bingogome/samm/blob/main/samm/SammBase/SammBaseLib/UtilMsgFactory.py#L233-L240, such that:
  ```python
  comboModelItems = ['vit_b', 'vit_l', 'vit_h', 'mobile_vit_t', 'medsam_vit_b', "medicosam_vit_b"]  # NEW MODEL NAME ADDITION
  ```
  - (OPTIONAL) Allow the model to fallback to CPU resources, if necessary, by changing the lines of code at https://github.com/bingogome/samm/blob/main/samm-python-terminal/utl_sam_server.py#L42-L48.
  ```python
  # Load the segmentation model
  if torch.cuda.is_available():
      self.device = "cuda"
      print("[SAMM INFO] CUDA detected. Waiting for Model ...")
  elif torch.backends.mps.is_available():
      self.device = "mps"
      print("[SAMM INFO] MPS detected. Waiting for Model ...")
  else:
      self.device = "cpu"
      print("[SAMM INFO] CPU detected. Waiting for Model ...")
  ```
  - Visit the documentation and watch the demo video for more details on annotation workflows.

### 4. [`SlicerSegmentWithSAM`](SlicerSegmentWithSAM) (Slicer-based annotation tool):

> Expects some code changes in the extension API.

- Install `SlicerSegmentWithSAM` from source or via slicer extension.
- Move the model checkpoints under your Slicer installation at `Slicer-X.X.X-linux-amd64/slicer.org/Extensions-XXXXX/SegmentWithSAM/lib/Slicer-X.X/<MODEL>.pt`
- Update the filepath to `self.modelCheckpoint` and the model choice to `self.modelVersion = "vit_h"` in https://github.com/mazurowski-lab/SlicerSegmentWithSAM/blob/main/SegmentWithSAM/SegmentWithSAM.py#L86, if using the scripts from source, or at `Slicer-X.X.X-linux-amd64/slicer.org/Extensions-XXXXX/SegmentWithSAM/lib/Slicer-X.X/qt-scripted-modules/SegmentWithSAM.py`
- Visit their documentation (https://github.com/mazurowski-lab/SlicerSegmentWithSAM?tab=readme-ov-file#usage) for more details on annotation workflows.
