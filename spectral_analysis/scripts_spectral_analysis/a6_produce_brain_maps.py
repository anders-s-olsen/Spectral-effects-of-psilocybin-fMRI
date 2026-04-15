import numpy as np
import pandas as pd
import json
import os
from spectral_analysis.helper_functions import plot_surf

with open('config.json', 'r') as f:
    config = json.load(f)

os.makedirs('figures/coef_map_parcel',exist_ok=True)
os.makedirs('figures/spectrum_map_parcel',exist_ok=True)

strategies = config['strategies']
strategies = ['9p']

######################################## logpower band/parcel, coefficient map ###############################################
p_value_threshold_logpower = 0.05/3
perm = '_perm' #if not permutation p-values, set perm=''
for strategy in strategies:
    df = pd.read_csv('data/results/logpower_stats_by_band_parcel_'+strategy+perm+'.csv')
    for covariate in config['covariates']:
        if covariate=='SDI':
            continue
        if covariate=='PPL_mcg_L':
            vmin = -0.15
            vmax = 0.15
        elif covariate=='SDI':
            vmin = -0.02
            vmax = 0.02
        for band in config['frequency_bands']:
            df_reduced = df[(df['covariate']==covariate) & (df['uncontrolled'] == band)]
            avg_map = df_reduced.groupby('controlled')['coefcovariate'].mean()
            mask = df_reduced.groupby('controlled')['pval'+perm].apply(lambda x: (x <= p_value_threshold_logpower).all())
            # avg_map.iloc[df_reduced['pval']>p_value_threshold] = np.nan
            avg_map[~mask] = np.nan
            plot_surf(avg_map, output_file='figures/coef_map_parcel/logpower_'+covariate+'_'+strategy+'_'+band, vmin=vmin, vmax=vmax)

# ######################################## Raw spectra band/parcel/interval, partial residuals ###############################################

# for strategy in strategies:
#     df = pd.read_csv('data/results/spectra_by_band_parcel_'+strategy+'.csv')
#     for band in config['frequency_bands']:
#         for time_interval in config['time_intervals']:
#             df_reduced = df[(df['time_interval'] == time_interval) & (df['band'] == band)]
#             avg_map = df_reduced.groupby('roi')['partial_residuals'].mean()
            
#             if band == 'slow-3':
#                 vmin = 4
#                 vmax = 12
#             elif band == 'slow-4':
#                 vmin = 7
#                 vmax = 15
#             elif band == 'slow-5':
#                 vmin = 10
#                 vmax = 18
#             plot_surf(avg_map, output_file='figures/spectrum_map_parcel/'+strategy+'_'+time_interval+'_'+band, vmin=vmin, vmax=vmax)

######################################## Entropy parcel, coefficient map ###############################################
p_value_threshold_entropy = 0.05
perm = '_perm' #if not permutation p-values, set perm=''
for strategy in strategies:
    df = pd.read_csv('data/results/entropy_by_parcel_'+strategy+'.csv')
    df_stats = pd.read_csv('data/results/entropy_stats_by_parcel_'+strategy+perm+'.csv')
    for covariate in config['covariates']:
        if covariate=='SDI':
            continue
        not_covariate = [c for c in config['covariates'] if c != covariate]
        df_covariate = df.drop(not_covariate, axis=1).dropna()   
        if covariate=='PPL_mcg_L':
            vmin = -0.01
            vmax = 0.01
        elif covariate=='SDI':
            vmin = -0.08
            vmax = 0.08
        pvals_df = df_stats[(df_stats['covariate']==covariate)]
        avg_map = pvals_df.groupby('controlled')['coefcovariate'].mean()
        # avg_map.iloc[pvals_df['pval']>p_value_threshold] = np.nan
        mask = pvals_df.groupby('controlled')['pval'+perm].apply(lambda x: (x <= p_value_threshold_entropy).all())
        avg_map[~mask] = np.nan
        plot_surf(avg_map, output_file='figures/coef_map_parcel/entropy_'+covariate+'_'+strategy, vmin=vmin, vmax=vmax)

# # ######################################## Entropy parcel/interval, partial residuals ###############################################

# for strategy in strategies:
#     df = pd.read_csv('data/results/entropy_by_parcel_'+strategy+'.csv')
#     for time_interval in config['time_intervals']:
#         df_reduced = df[(df['time_interval'] == time_interval)]
#         avg_map = df_reduced.groupby('roi')['partial_residuals'].mean()
        
#         vmin = 8
#         vmax = 8.5
#         plot_surf(avg_map, output_file='figures/spectrum_map_parcel/entropy_'+strategy+'_'+time_interval, vmin=vmin, vmax=vmax)
