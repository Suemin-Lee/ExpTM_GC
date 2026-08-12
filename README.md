# Exponentially Tilted Thermodynamic Maps (expTM)

This repository contains the implementation and results from the paper
**[Exponentially Tilted Thermodynamic Maps (expTM): Predicting Phase Transitions Across Temperature, Pressure, and Chemical Potential](https://arxiv.org/abs/2503.15080)**
applied to the **Lattice Gas model** in the grand canonical (μVT) ensemble.

👉 Try it out on **Google Colab**:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q5-LyumxZfZc6aHt71ZKOsGWfjrGEEzV?usp=sharing)

### Lattice Gas (μVT)

The grand canonical implementation uses temperature (T) and chemical potential (\mu) as thermodynamic control variables. The main workflow is provided in `GC_original.ipynb`, `GC_original.py`, and `GC_ising_code.ipynb`.

### CO₂ (NPT)

The NPT application uses temperature (T) and pressure (P) as thermodynamic control variables to generate ensembles and study pressure-driven phase behavior. Pressure-dependent data, trained models, and analysis outputs are contained within the corresponding `data/`, `models/`, and `plots/` directories.

### Core expTM Framework

The `tm/` directory contains the core Thermodynamic Maps implementation used by the different thermodynamic ensembles and applications.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone git@github.com:Suemin-Lee/ExpTM_GC.git
cd ExpTM_GC/thermomaps-root
pip install -r requirements.txt
```

## Citation

If you use this code, please cite:
DOI:** *To be updated upon publication.*

```bibtex
@misc{lee2025exponentiallytiltedthermodynamicmaps,
    title={Exponentially Tilted Thermodynamic Maps (expTM): Predicting Phase Transitions Across Temperature, Pressure, and Chemical Potential},
    author={Suemin Lee and Ruiyu Wang and Lukas Herron and Pratyush Tiwary},
    year={2025},
    eprint={2503.15080},
    archivePrefix={arXiv},
    primaryClass={cond-mat.stat-mech},
    url={https://arxiv.org/abs/2503.15080}
}
```
