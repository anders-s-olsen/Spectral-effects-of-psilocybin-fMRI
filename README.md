# Spectral Effects of Psilocybin on fMRI Signals

This repository accompanies the preprint:

👉 https://www.biorxiv.org/content/10.64898/2026.04.09.717379v1

## Overview

This repository contains the analysis code used to investigate the spectral effects of psilocybin on fMRI signals. It is made publicly available to ensure transparency and reproducibility of the methods described in the associated manuscript.

⚠️ **Important:**  
This repository is **not intended to function as a standalone toolbox**. The scripts depend on data that are **not included** (e.g., `data/` directory), and therefore cannot be executed directly without access to the original datasets and preprocessing outputs.

## Purpose

- Provide full methodological transparency for peer review and publication  
- Document the computational workflow used in the study  
- Enable qualified researchers to reproduce or adapt the analysis (given appropriate data access)  

## Repository Structure
spectral_analysis/
│
├── helper_functions.py
│
├── scripts_preproc/
│ ├── p1_make_table_of_func_scans.py
│ ├── p2_extract_confounds_outliers.py
│ ├── p3_align_func_to_ppl_sdi.py
│ ├── p4_denoise_timeseries_voxelwise.py
│ ├── p5_compute_tSNR.py
│ └── p6_parcellate_timeseries.py
│
├── scripts_spectral_compute/
│ ├── s1_compute_mtspectra_voxelwise.py
│ ├── s2_compute_mtspectra_motion.py
│ ├── s3_compute_spectral_entropy_voxelwise.py
│ └── s4_parcellate_spectra_and_entropy.py
│
├── scripts_spectral_analysis/
│ ├── a1_prepare_df_frequencies_networks.py
│ ├── a2_prepare_df_bands_parcels.py
│ ├── a3_prepare_df_entropy_parcels.py
│ ├── a5_stats_perm_maxT_R.R
│ └── a6_produce_brain_maps.py
│
├── scripts_connectivity/
│ ├── c1_compute_connectomes_parcels.py
│ ├── c2_prepare_df_bands.py
│ └── c3_prepare_df_bands_networks.py
│
├── scripts_ged/
│ ├── g1_compute_ged.py
│ ├── g2_prepare_df_frequencies_eigenvectors.py
│ └── g3_prepare_df_rayleigh.py
│
└── visualization/
├── notebooks for figures and exploration

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

### 4. Connectivity & GED
- Functional connectivity estimation  
- Network-level summaries  
- Generalized eigendecomposition (GED) analyses  

### 5. Visualization
- Jupyter notebooks for reproducing figures and exploratory analyses  

## Installation

A `setup.py` file is included, but installation is **optional** and not required for understanding the workflow.

If desired:

```bash
pip install -e .
