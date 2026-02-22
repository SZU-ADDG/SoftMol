---
library_name: pytorch
tags:
- chemistry
- biology
- drug-discovery
- molecular-generation
- diffusion
license: mit
language: en
datasets:
- SZU-ADDG/ZINC-Curated
---

<div align="center">
  <img src="https://huggingface.co/SZU-ADDG/SoftMol/resolve/main/image/overview.png" width="800"/>
</div>

## Model Description

**SoftMol** is a unified framework for target-aware molecular generation that systematically co-designs representation, model architecture, and search strategy. It introduces the **SoftBD (Soft-fragment Block-Diffusion)** architecture, which is the first molecular block-diffusion language model. SoftMol synergizes intra-block bidirectional denoising with inter-block autoregressive conditioning.

## Model Sources

- **Repository:** [GitHub Repository](https://github.com/szu-aicourse/softmol) (Please check our GitHub for full codebase)
- **Paper:** [arXiv Paper Link](https://arxiv.org/abs/2601.21964)

## Available Model Weights

We provide checkpoints for multiple model scales. The **89M** model (`89M-epoch6-best.ckpt`) is the primary checkpoint used for the results reported in the paper.

- `55M-epoch1-last.ckpt` (config: `small-50M.yaml`)
- `74M-epoch1-last.ckpt` (config: `small-70M.yaml`)
- `89M-epoch6-best.ckpt` (config: `small-89M.yaml`) **[Recommended]**
- `116M-epoch1-last.ckpt` (config: `small-110M.yaml`)
- `624M-epoch1-last.ckpt` (config: `large.yaml`)

## Training Details

### Training Data
SoftMol is trained on the **ZINC-Curated** dataset, a carefully curated collection of molecules favored for high drug-likeness and synthetic accessibility. The dataset is available at [SZU-ADDG/ZINC-Curated](https://huggingface.co/datasets/SZU-ADDG/ZINC-Curated).

### Hardware & Hyperparameters (89M Model)
- **Hardware:** 8 × NVIDIA RTX 4090 GPUs
- **Precision:** `bf16-mixed`
- **Global Batch Size:** 1600
- **Attention Backend:** SDPA
- **Steps:** 1,334,000

## Intended Use & Capabilities

### 1. De Novo Generation
For unconstrained molecule generation, SoftMol (SoftBD) can generate chemically valid and diverse molecules efficiently. In our experiments (using $K_{\text{sample}}=2$, $p=0.95$, $\tau=1.0$), SoftBD achieved **100% chemical validity**.

### 2. Structure-Based Drug Design (SBDD)
SoftMol can generate ligands for specific protein targets (`parp1`, `jka2`, `fa7`, `5ht1b`, `braf`) utilizing a **gated MCTS (Monte Carlo Tree Search)** mechanism. This mechanism explicitly decouples binding affinity optimization from drug-likeness constraints.

## Performance & Results

Empirically, SoftMol resolves the trade-off between generation quality and efficiency. Compared to state-of-the-art methods:
- **100% chemical validity**
- **6.6x speedup** in inference efficiency
- **9.7% improvement** in binding affinity
- **2-3x higher** molecular diversity 

## How to Get Started with the Model

To use this model, please clone our [GitHub repository](https://github.com/szu-aicourse/softmol) and set up the environment.

**Download weights dynamically:**
```python
from huggingface_hub import hf_hub_download

# Download the recommended 89M weight
model_path = hf_hub_download(repo_id="SZU-ADDG/SoftMol", filename="weights/89M-epoch6-best.ckpt")

# Download the corresponding config
config_path = hf_hub_download(repo_id="SZU-ADDG/SoftMol", filename="configs/model/small-89M.yaml")
```

## Citation

If you use SoftMol or the ZINC-Curated dataset in your research, please cite our paper:

```bibtex
@article{yang2026tokens,
  title={From Tokens to Blocks: A Block-Diffusion Perspective on Molecular Generation},
  author={Yang, Qianwei and Xu, Dong and Yang, Zhangfan and Yuan, Sisi and Zhu, Zexuan and Li, Jianqiang and Ji, Junkai},
  journal={arXiv preprint arXiv:2601.21964},
  year={2026}
}
```
