<div align="center">

<h1 align="center">
  <img src="logo.png" alt="logo" width="40" />
  MambaX: Image Super-Resolution with State Predictive Control
</h1>

Chenyu Li, [Danfeng Hong](https://scholar.google.com/citations?hl=en&user=n7gL0_IAAAAJ&view_op=list_works&sortby=pubdate), Bing Zhang, et al.
 
[https://doi.org/10.1109/TPAMI.2026.3721958](https://doi.org/10.1109/TPAMI.2026.3721958)
</div>

## 📄 Abstract
Image super-resolution (SR) is a critical technology for overcoming the inherent hardware limitations of sensors. However, existing approaches mainly focus on directly enhancing the final resolution, often neglecting effective control over error propagation and accumulation during intermediate stages. Recently, Mamba has emerged as a promising approach that can represent the entire reconstruction process as a state sequence with multiple nodes, allowing for intermediate intervention. Nonetheless, its fixed linear mapper is limited by a narrow receptive field and restricted flexibility, which hampers its effectiveness in fine-grained images. To address this, we created a nonlinear state predictive control model MambaX that maps consecutive spectral bands into a latent state space and generalizes the SR task by dynamically learning the nonlinear state parameters of control equations. Compared to existing sequence models, MambaX 1) employs dynamic state predictive control learning to approximate the nonlinear differential coefficients of state-space models; 2) introduces a novel state cross-control paradigm for multimodal SR fusion; and 3) utilizes progressive transitional learning to mitigate heterogeneity caused by domain and modality shifts. Our evaluation demonstrates the superior performance of the dynamic spectrum-state representation model in both single-image SR and multimodal fusion-based SR tasks, highlighting its substantial potential to advance spectrally generalized modeling across arbitrary dimensions and modalities.


## 👀 Overview

![alt text](./Framework.png)


## 🛠️ Environment Preparation

Install Python dependencies by running:

```bash
pip install -r requirements.txt

```

and set up the environment by running:

```bash
bash install.sh
```

## 📋 Data Preparation

Datasets are placed under the repository-level `data/` directory.

- **Single-Modal Data**

>  You may need to manually download the [CAVE](https://pan.baidu.com/s/15mB03bdToGivx0SKhdMo9g) with `vbp9`, [Pavia](https://pan.baidu.com/s/1BHKiuGZZ_4ZpWBvnuiREHQ) with `xz9h`, and [Chikusei](https://pan.baidu.com/s/1bQayqSjDOImWEz_CZ4GDaw) with `nkq8`.

>  For example, Pavia ×2 should be organized as:

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

The same convention is used for CAVE and Chikusei at ×2, ×4, and ×8 scales.

- **Multi-Modal Data**

>  You may need to manually download the [GF2](https://pan.baidu.com/s/1p-TnQK_khEuxyMNPAgWCQQ) with `eykq`, and [WV3](https://pan.baidu.com/s/1hZy5yBJ-W-3n4p2zuFANdg) with `ftjd`.

>  The expected multimodal data structure is:
 
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


## 📝 Citation

If you find our project helpful, please cite our paper:

C. Li, D. Hong, B. Zhang, et, al. "MambaX: Image Super-Resolution with State Predictive Control," IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2026.3721958.

```bibtex
@ARTICLE{11646487,
  author={Li, Chenyu and Hong, Danfeng and Zhang, Bing and Pan, Zhaojie and Yokoya, Naoto and Chanussot, Jocelyn},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={MambaX: Image Super-Resolution with State Predictive Control}, 
  year={2026},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TPAMI.2026.3721958}}
```

---

## 📜 Licensing

Copyright © 2026 Danfeng Hong

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.

---

## 📧 Contact Information

**Danfeng Hong**: hongdanfeng1989@gmail.com  
School of Automation, Southeast University, 211189 Nanjing, China.

---
