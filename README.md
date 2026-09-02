<div align="center">
<h1>MambaX: Image Super-Resolution with State Predictive Control 
</h1> 

Chenyu Li, [Danfeng Hong](https://scholar.google.com/citations?hl=en&user=n7gL0_IAAAAJ&view_op=list_works&sortby=pubdate), Bing Zhang, et al.

([https://ieeexplore.ieee.org/document/11646487](https://ieeexplore.ieee.org/document/11646487))
</div>

## 📄 Abstract
Image super-resolution (SR) is a critical technology for overcoming the inherent hardware limitations of sensors. However, existing approaches mainly focus on directly enhancing the final resolution, often neglecting effective control over error propagation and accumulation during intermediate stages. Recently, Mamba has emerged as a promising approach that can represent the entire reconstruction process as a state sequence with multiple nodes, allowing for intermediate intervention. Nonetheless, its fixed linear mapper is limited by a narrow receptive field and restricted flexibility, which hampers its effectiveness in fine-grained images. To address this, we created a nonlinear state predictive control model MambaX that maps consecutive spectral bands into a latent state space and generalizes the SR task by dynamically learning the nonlinear state parameters of control equations. Compared to existing sequence models, MambaX 1) employs dynamic state predictive control learning to approximate the nonlinear differential coefficients of state-space models; 2) introduces a novel state cross-control paradigm for multimodal SR fusion; and 3) utilizes progressive transitional learning to mitigate heterogeneity caused by domain and modality shifts. Our evaluation demonstrates the superior performance of the dynamic spectrum-state representation model in both single-image SR and multimodal fusion-based SR tasks, highlighting its substantial potential to advance spectrally generalized modeling across arbitrary dimensions and modalities.

---
## 👀 Overview

![alt text](./Framework.png)

---



The repository contains the code required for two image super-resolution settings:

- **Single-modal hyperspectral image super-resolution**, including CAVE, Pavia, and Chikusei.
- **Multimodal fusion super-resolution**, including WorldView-III (WV3) and GaoFen-2 (GF2).

---



## Repository Structure

The main entry points are:

| Task | Entry |
|---|---|
| Single-modal training/testing | `single_modal/mains.py` |
| WV3 multimodal training | `multimodal/train_w.py` |
| GF2 multimodal training | `multimodal/train_g.py` |
| Multimodal reduced-resolution testing | `multimodal/test.py` |

---

## Environment

The following environment was used for the full reproduction experiments:

| Package | Version |
|---|---|
| Python | 3.9.18 |
| PyTorch | 1.13.1+cu117 |
| CUDA used by PyTorch | 11.7 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| h5py | 3.11.0 |
| OpenCV | 4.10.0 |
| einops | 0.8.0 |
| thop | 0.1.1 |

A Linux system with an NVIDIA CUDA-capable GPU is recommended because the Mamba selective-scan operators use CUDA extensions.

### 1. Create a Python environment

```bash
conda create -n mambax python=3.9.18 -y
conda activate mambax
```

### 2. Install PyTorch

For the tested CUDA 11.7 configuration:

```bash
pip install \
  torch==1.13.1+cu117 \
  torchvision==0.14.1+cu117 \
  torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

### 3. Install Python dependencies

```bash
pip install \
  numpy==1.26.4 \
  scipy==1.13.1 \
  h5py==3.11.0 \
  opencv-python==4.10.0.84 \
  einops==0.8.0 \
  thop \
  tensorboardX \
  torchnet \
  pytorch-msssim \
  packaging \
  ninja
```

### 4. Build the bundled Mamba CUDA extensions

The required Mamba and causal-conv1d source code is included under `multimodal/`.

From the repository root:

```bash
cd multimodal/causal-conv1d

rm -rf build *.egg-info
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .

cd ../mamba

rm -rf build *.egg-info
MAMBA_FORCE_BUILD=TRUE pip install .

cd ../..
```

The bundled source currently reports:

```text
causal-conv1d: 1.0.1
mamba-ssm:     1.1.1
```

After installation, the core selective-scan operator can be checked with:

```bash
python -c "
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('selective_scan_fn: OK')
"
```

CUDA extension compilation depends on the local PyTorch, CUDA toolkit, compiler, and GPU environment. If rebuilding the extensions, make sure the CUDA toolkit used by the compiler is compatible with the installed PyTorch CUDA build.

---

# Data Preparation

The datasets are **not included** in this repository.

The default code assumes that datasets are placed under the repository-level `data/` directory.

---

## Single-Modal Data

Supported datasets:

```text
CAVE
Pavia
Chikusei
```

The default directory rule is:

```text
data/single_modal/
└── <dataset_lower>_2/
    └── <dataset_name>_x<scale>/
        ├── trains/
        ├── evals/
        └── <dataset_name>_test.mat
```

For example, Pavia ×2 should be organized as:

```text
data/single_modal/
└── pavia_2/
    └── pavia_x2/
        ├── trains/
        │   └── *.mat
        ├── evals/
        │   └── *.mat
        └── pavia_test.mat
```

The same convention is used for ×4 and ×8 experiments.

For Chikusei, use `Chikusei` as the dataset name when calling the script.

---

## Multimodal Data

The expected multimodal data structure is:

```text
data/multimodal/
├── training_wv3/
│   ├── train_wv3.h5
│   ├── valid_wv3.h5
│   └── reduced_examples/
│       └── test_wv3_multiExm1.h5
│
└── training_gf2/
    ├── train_gf2.h5
    ├── valid_gf2.h5
    └── reduced_examples/
        └── test_gf2_multiExm1.h5
```

Each H5 file uses the following keys:

```text
gt
lms
ms
pan
```

### WV3

Training and validation samples use:

```text
GT/LMS : 8 × 64 × 64
MS     : 8 × 16 × 16
PAN    : 1 × 64 × 64
```

Reduced-resolution test samples use:

```text
GT/LMS : 8 × 256 × 256
MS     : 8 × 64 × 64
PAN    : 1 × 256 × 256
```

The WV3 loader normalizes the data by `2047`.

### GF2

Training and validation samples use:

```text
GT/LMS : 4 × 64 × 64
MS     : 4 × 16 × 16
PAN    : 1 × 64 × 64
```

Reduced-resolution test samples use:

```text
GT/LMS : 4 × 256 × 256
MS     : 4 × 64 × 64
PAN    : 1 × 256 × 256
```

The GF2 loader normalizes the data by `1023`.

---

# Single-Modal Super-Resolution

Change to the single-modal directory:

```bash
cd single_modal
```

The main script is:

```text
mains.py
```

Use:

```bash
python mains.py --help
```

to display all configurable arguments.

## Training

Example: Pavia ×2

```bash
CUDA_VISIBLE_DEVICES=0 \
python mains.py \
  --subcommand train \
  --cuda 1 \
  --dataset_name pavia \
  --n_scale 2 \
  --n_feats 256 \
  --epochs 150 \
  --learning_rate 0.0003 \
  --gpus 0
```

Example: CAVE ×2

```bash
CUDA_VISIBLE_DEVICES=0 \
python mains.py \
  --subcommand train \
  --cuda 1 \
  --dataset_name cave \
  --n_scale 2 \
  --n_feats 256 \
  --epochs 150 \
  --learning_rate 0.0001 \
  --gpus 0
```

Example: Chikusei ×2

```bash
CUDA_VISIBLE_DEVICES=0 \
python mains.py \
  --subcommand train \
  --cuda 1 \
  --dataset_name Chikusei \
  --n_scale 2 \
  --n_feats 256 \
  --epochs 150 \
  --learning_rate 0.0003 \
  --gpus 0
```

For ×4 or ×8 experiments, change:

```text
--n_scale 4
```

or:

```text
--n_scale 8
```

The paper reports 150 training epochs for the single-modal experiments. The initial learning rate is `1e-4` for CAVE and `3e-4` for Pavia and Chikusei, with the learning rate halved every 30 epochs.

## Testing

A checkpoint must be provided explicitly.

Example: Pavia ×2

```bash
CUDA_VISIBLE_DEVICES=0 \
python mains.py \
  --subcommand test \
  --cuda 1 \
  --dataset_name pavia \
  --n_scale 2 \
  --n_feats 256 \
  --checkpoint /path/to/model.pth \
  --gpus 0
```

Replace `/path/to/model.pth` with the checkpoint to be evaluated.

The test procedure computes hyperspectral reconstruction metrics and saves reconstructed outputs for further analysis.

---

# Multimodal Fusion Super-Resolution

Change to:

```bash
cd multimodal
```

The multimodal training code uses Adam optimization and the HLoss objective implemented in `utilities/utils.py`.

The paper reports 500 epochs for multimodal fusion experiments with an initial learning rate of `7e-4`.

---

## WV3 Training

The official WV3 model configuration used by this repository is:

```text
method = SDL_ms
band   = 8
dim    = 32
```

With the default dataset structure, training can be started with:

```bash
CUDA_VISIBLE_DEVICES=0 \
python train_w.py \
  --gpu_id 0 \
  --method SDL_ms \
  --band 8 \
  --dim 32 \
  --max_epoch 500 \
  --learning_rate 0.0007
```

The default data locations are:

```text
data/multimodal/training_wv3/train_wv3.h5
data/multimodal/training_wv3/valid_wv3.h5
data/multimodal/training_wv3/reduced_examples/test_wv3_multiExm1.h5
```

Custom paths can be supplied using:

```text
--data_path_train
--data_path_test
--reduced_test_path
--outf
```

---

## GF2 Training

The official GF2 model configuration used by this repository is:

```text
method = SDL_msg
band   = 4
dim    = 24
```

Start training with:

```bash
CUDA_VISIBLE_DEVICES=0 \
python train_g.py \
  --gpu_id 0 \
  --method SDL_msg \
  --band 4 \
  --dim 24 \
  --max_epoch 500 \
  --learning_rate 0.0007
```

The default data locations are:

```text
data/multimodal/training_gf2/train_gf2.h5
data/multimodal/training_gf2/valid_gf2.h5
data/multimodal/training_gf2/reduced_examples/test_gf2_multiExm1.h5
```

---

# Multimodal Testing

The cleaned test entry is:

```text
multimodal/test.py
```

It supports both WV3 and GF2 reduced-resolution evaluation.

## WV3

From the `multimodal/` directory:

```bash
CUDA_VISIBLE_DEVICES=0 \
python test.py \
  --dataset wv3 \
  --weight checkpoints/wv3/model.pth \
  --device cuda:0
```

## GF2

```bash
CUDA_VISIBLE_DEVICES=0 \
python test.py \
  --dataset gf2 \
  --weight checkpoints/gf2/model.pth \
  --device cuda:0
```

The default reduced-resolution H5 file is selected automatically according to `--dataset`.

A custom test file can be specified with:

```bash
--file_path /path/to/test.h5
```

A custom output directory can be specified with:

```bash
--save_dir /path/to/output
```

For `test.py`, relative paths supplied through the command line are interpreted relative to the **repository root**, not relative to the `multimodal/` directory.

The multimodal training scripts save complete PyTorch model objects using `torch.save(model, ...)`. Therefore, the test script expects a compatible complete-model checkpoint. PyTorch version differences can affect loading of serialized full-model checkpoints.

---

# Output Files

Multimodal training outputs are written under:

```text
outputs/multimodal/
```

with separate WV3 and GF2 experiment directories.

A typical experiment contains:

```text
<timestamp>/
├── result/
│   └── output.txt
└── model/
    ├── model_001.pth
    ├── model_002.pth
    └── ...
```

The multimodal test script saves reconstructed samples as `.mat` files.

Runtime outputs, logs, datasets, and model checkpoints are excluded from Git tracking through `.gitignore`.

---

# Paper

**MambaX: Image Super-Resolution with State Predictive Control**

Paper publication information, project page, and formal citation can be added here.
