# Spectral Effects of Psilocybin on fMRI Signals

This repository accompanies the preprint:

👉 https://www.biorxiv.org/content/10.64898/2026.04.09.717379v1

## Overview

This repository contains the analysis code used to investigate the spectral effects of psilocybin on fMRI signals. It is made publicly available to ensure transparency and reproducibility of the methods described in the associated manuscript.

⚠️ **Important:**  
This repository is **not intended to function as a standalone toolbox**. The scripts depend on data that are **not included** (e.g., `data/` directory), and therefore cannot be executed directly without access to the original datasets and preprocessing outputs.

## Workflow Summary

The analysis pipeline consists of several major stages:

### 1. Preprocessing
- Functional scan organization  
- Confound extraction  
- Denoising and alignment  
- Time series parcellation  

### 2. Spectral Computation
- Multitaper spectral estimation  
- Spectral entropy computation  
- Parcel-level aggregation  

### 3. Spectral Analysis
- Frequency- and band-level summaries  
- Statistical testing (including permutation methods in R)  
- Brain map generation  

### 4. Visualization
- Jupyter notebooks for reproducing figures and exploratory analyses  

## Installation

A `setup.py` file is included, but installation is **optional** and not required for understanding the workflow.

If desired:

```bash
pip install -e .
```

## Data Availability

The data required to run these scripts are not included in this repository.

This includes (but is not limited to):

- Raw and preprocessed fMRI data
- Confound regressors
- Intermediate outputs stored in the data/ directory

Access to these data is subject to the policies described in the associated publication.

## Reproducibility Notes
Scripts are designed to be executed in sequence within each pipeline stage
File paths and configurations are controlled via config.json
Some statistical analyses rely on R scripts included in the repository

## Citation

If you use or refer to this code, please cite the preprint:

https://www.biorxiv.org/content/10.64898/2026.04.09.717379v1

## Disclaimer

This repository is provided for methodological transparency only.
It is not maintained as a general-purpose software package, and limited support should be expected.
