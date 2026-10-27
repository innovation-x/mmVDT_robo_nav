# mmVDT for Mobile-Robot Navigation

This is the implementation and qualitative results for the paper [“Action-Inclusive Multi-Future Prediction Using a Generative Model in Human-Related Scenes for Mobile Robots”](https://ieeexplore.ieee.org/document/11170425/metrics#full-text-header).

mmVDT is a multimodal diffusion-transformer model that jointly models future visual observations and robot motion. The repository includes RGB and panoptic/semantic inputs, VAE training, single- and multi-GPU diffusion training, and notebooks for qualitative and quantitative evaluation.

## Repository contents

- `TimeSformer/`: mmVDT model variants.
- `DDIM/`: diffusion sampling utilities.
- `dataset/`: dataset loader (datasets themselves are intentionally not versioned).
- `train_vae.py`, `train_VDT_*.py`: VAE and mmVDT training entry points.
- `eval/`, `eval_3c/`, and `evaluation*.ipynb`: metrics and analysis notebooks.
- `VDT_cat*.ipynb` and `data_generator.ipynb`: interactive inference and data-generation workflows.
- `MaskDINO/`: pinned third-party segmentation dependency, provided as a Git submodule.
- `assets/results/`: qualitative examples used below.

## Setup

Use Python 3.8 and a CUDA-compatible PyTorch installation. Clone with the segmentation dependency, then install the Python packages:

```bash
git clone --recurse-submodules https://github.com/innovation-x/mmVDT_robo_nav.git
cd mmVDT_robo_nav
pip install -r requirements.txt
pip install -r MaskDINO/requirements.txt
```

MaskDINO additionally requires Detectron2 and compilation of its CUDA extension; follow its [installation guide](MaskDINO/INSTALL.md) after initializing the submodule. Checkpoints, datasets, experiment logs, and generated metric tensors are excluded from Git because they are large, machine-specific artifacts.

## Results

The following examples compare observed sequences from the dataset with multiple future trajectories sampled by the model. The semantic and RGB result assets are incorporated from the companion [mmVDT result gallery](https://github.com/innovation-x/mmVDT_for_mobile_robot).

### Panoptic / semantic inputs

| Sample | Observed sequence | Generated future 1 | Generated future 2 | Generated future 3 |
| --- | --- | --- | --- | --- |
| 1 | ![Semantic real sample 1](assets/results/semantic/real_data_videos/sample_1.gif) | ![Semantic generated sample 1-1](assets/results/semantic/generated_videos/sample_1_1.gif) | ![Semantic generated sample 1-2](assets/results/semantic/generated_videos/sample_1_2.gif) | ![Semantic generated sample 1-3](assets/results/semantic/generated_videos/sample_1_3.gif) |
| 2 | ![Semantic real sample 2](assets/results/semantic/real_data_videos/sample_2.gif) | ![Semantic generated sample 2-1](assets/results/semantic/generated_videos/sample_2_1.gif) | ![Semantic generated sample 2-2](assets/results/semantic/generated_videos/sample_2_2.gif) | ![Semantic generated sample 2-3](assets/results/semantic/generated_videos/sample_2_3.gif) |
| 3 | ![Semantic real sample 3](assets/results/semantic/real_data_videos/sample_3.gif) | ![Semantic generated sample 3-1](assets/results/semantic/generated_videos/sample_3_1.gif) | ![Semantic generated sample 3-2](assets/results/semantic/generated_videos/sample_3_2.gif) | ![Semantic generated sample 3-3](assets/results/semantic/generated_videos/sample_3_3.gif) |

### RGB inputs

| Sample | Observed sequence | Generated future 1 | Generated future 2 | Generated future 3 |
| --- | --- | --- | --- | --- |
| 1 | ![RGB real sample 1](assets/results/rgb/real_data_videos/sample_1.gif) | ![RGB generated sample 1-1](assets/results/rgb/generated_videos/sample_1_1.gif) | ![RGB generated sample 1-2](assets/results/rgb/generated_videos/sample_1_2.gif) | ![RGB generated sample 1-3](assets/results/rgb/generated_videos/sample_1_3.gif) |
| 2 | ![RGB real sample 2](assets/results/rgb/real_data_videos/sample_2.gif) | ![RGB generated sample 2-1](assets/results/rgb/generated_videos/sample_2_1.gif) | ![RGB generated sample 2-2](assets/results/rgb/generated_videos/sample_2_2.gif) | ![RGB generated sample 2-3](assets/results/rgb/generated_videos/sample_2_3.gif) |
| 3 | ![RGB real sample 3](assets/results/rgb/real_data_videos/sample_3.gif) | ![RGB generated sample 3-1](assets/results/rgb/generated_videos/sample_3_1.gif) | ![RGB generated sample 3-2](assets/results/rgb/generated_videos/sample_3_2.gif) | ![RGB generated sample 3-3](assets/results/rgb/generated_videos/sample_3_3.gif) |

## Third-party software

The `MaskDINO` submodule remains under its own license. See [its repository](https://github.com/IDEA-Research/MaskDINO) and included license information for details.
