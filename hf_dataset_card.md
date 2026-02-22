---
license: mit
task_categories:
- text-generation
tags:
- chemistry
- biology
- drug-discovery
- computational-chemistry
- smiles
size_categories:
- 100M<n<1B
---

# Dataset Card for ZINC-Curated

## Dataset Description

- **Repository:** [SZU-ADDG/SoftMol](https://github.com/SZU-ADDG/SoftMol)
- **Paper:** [From Tokens to Blocks: A Block-Diffusion Perspective on Molecular Generation](https://arxiv.org/abs/2601.21964)

### Dataset Summary

**ZINC-Curated** is a high-quality, large-scale dataset of approximately 427 million drug-like molecules in SMILES format. To prioritize pharmaceutical relevance, it was constructed by curating a high-quality subset of the ZINC-22 database using a rigorous multi-stage filtration pipeline. 

The pipeline enforces stringent physicochemical constraints, structural validity, medicinal chemistry rules, and diversity-aware stratification, making it an ideal pre-training corpus for large-scale molecular language models and generative de novo drug design.

## Loading the Dataset

This dataset can be directly loaded and used via the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("SZU-ADDG/ZINC-Curated")

# Access splits
train_data = dataset["train"]
valid_data = dataset["validation"]
```

## Dataset Structure

The corpus contains over 427M valid SMILES strings meticulously stored in scalable Arrow format. 

### Data Instances
A typical instance in the dataset is a single SMILES string representing a target drug-like molecule:
```python
{'input': 'CC1=CC=C(C=C1)S(=O)(=O)N(C)C2=CC=CC=C2'}
```

## Dataset Creation & Preprocessing

The dataset was constructed by collecting molecules from [ZINC-22](https://zinc22.docking.org/) and applying a four-stage curation pipeline adapted from [NovoMolGen](https://arxiv.org/abs/2508.13408) to ensure pharmaceutical relevance and structural diversity:

1. **Physicochemical Filtering:**
   Discarded molecules with Quantitative Estimate of Drug-likeness (QED) <= 0.5 or Synthetic Accessibility (SA) >= 5, which correspond to poor drug-likeness or low synthetic accessibility.

2. **Structural Validity:**
   Removed compounds containing undesirable elements (e.g., Si or Sn), carrying non-neutral charges, including free radicals, or exhibiting overly complex topologies (e.g., more than two bridgehead atoms, rings larger than eight members, or more than ten rotatable bonds). Molecules with Molecular Polar Surface Area (TPSA) > 140 or known toxic/PAINS substructures were also explicitly excluded.

3. **Medicinal Chemistry Rules:**
   Imposed Lipinski’s Rule of Five, constraining:
   - Molecular Weight (MW): 100 <= MW <= 500
   - Lipophilicity (LogP): LogP <= 5
   - Strict limitations on hydrogen bond donor and acceptor counts.

4. **Diversity-Aware Stratification:**
   To mitigate dataset bias, remaining molecules were grouped by heavy-atom count between 4 and 49. Within each bucket, we retained only those whose Tanimoto similarity to previously accepted molecules was strictly below 0.5.

5. **Sequence Length Constraint:**
   We imposed a maximum SMILES sequence length of L = 72 tokens. The tokenization length is calculated specifically using our custom SMILES vocabulary ([vocab_V2.txt](https://github.com/SZU-ADDG/SoftMol/blob/main/vocab_V2.txt)). This constraint removed a negligible amount of data (< 0.001%) while retaining approximately 427 million high-quality molecules.

## Citation

If you find this dataset useful in your research, please cite our corresponding paper:

```bibtex
@article{yang2026tokens,
  title={From Tokens to Blocks: A Block-Diffusion Perspective on Molecular Generation},
  author={Yang, Qianwei and Xu, Dong and Yang, Zhangfan and Yuan, Sisi and Zhu, Zexuan and Li, Jianqiang and Ji, Junkai},
  journal={arXiv preprint arXiv:2601.21964},
  year={2026}
}
```
