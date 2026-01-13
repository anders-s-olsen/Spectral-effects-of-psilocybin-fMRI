import nibabel as nib
import nilearn.surface as surface
import nilearn.plotting as plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
from subcortex_visualization import plot_subcortical_data
from spectral_analysis.helper_functions import import_mask_and_parcellation
from scipy.stats import chi2

with open('config.json', 'r') as f:
    config = json.load(f)

p_value_threshold = 0.05/3

parcel_labels, parcellation, masks, parcellation_extended = import_mask_and_parcellation()
unique_parcels = np.unique(parcellation)
unique_parcels = unique_parcels[unique_parcels != 0]  # remove background parcel

# ######################################## Entropy, coefficient map in parcels ###############################################
# os.makedirs('figures/coef_map_parcel',exist_ok=True)
# for strategy in config['strategies']:
#     df = pd.read_csv('data/results/entropy_by_parcel_'+strategy+'.csv')
#     df_stats = pd.read_csv('data/results/entropy_stats_by_parcel_'+strategy+'_perm.csv')
#     for covariate in config['covariates']:
#         if covariate=='SDI':
#             continue
#         not_covariate = [c for c in config['covariates'] if c != covariate]
#         df_covariate = df.drop(not_covariate, axis=1).dropna()   
#         if covariate=='PPL_mcg_L':
#             vmin = -14
#             vmax = 14
#         elif covariate=='SDI':
#             vmin = -0.08
#             vmax = 0.08
#         pvals_df = df_stats[(df_stats['covariate']==covariate)]
#         avg_map = pvals_df.groupby('controlled')['coefcovariate'].mean()
#         # avg_map.iloc[pvals_df['pval']>p_value_threshold] = np.nan
#         mask = pvals_df.groupby('controlled')['pval_perm'].apply(lambda x: (x <= p_value_threshold).all())
#         avg_map[~mask] = np.nan
#         for hemi in ['lh', 'rh']:
#             print(f'Visualizing {covariate} {strategy} entropy {hemi}')
            
#             if hemi=='lh':
#                 cortex_map = np.empty(masks['mask_lh'].shape)
#                 cortex_map.fill(np.nan)
#                 for roi_label, roi_number in zip(parcel_labels,unique_parcels):
#                     if 'LH' in roi_label:
#                         cortex_map[parcellation_extended['parcellation_lh_expanded']==roi_number] = avg_map[roi_label]
#                 h = 'L'
#                 h2 = 'left'
#             elif hemi=='rh':
#                 cortex_map = np.empty(masks['mask_rh'].shape)
#                 cortex_map.fill(np.nan)
#                 for roi_label, roi_number in zip(parcel_labels,unique_parcels):
#                     if 'RH' in roi_label:
#                         cortex_map[parcellation_extended['parcellation_rh_expanded']==roi_number] = avg_map[roi_label]
#                 h = 'R'
#                 h2 = 'right'
#             fsLR_surface = surface.load_surf_mesh('data/external/fs_LR.32k.'+h+'.midthickness.surf.gii')
#             for view in ['lateral','medial']:
#                 plotting.plot_surf(fsLR_surface,cortex_map,symmetric_cmap=None,bg_on_data=False,colorbar=True, 
#                                     cmap='bwr', vmin=vmin, vmax=vmax, darkness=None, hemi=h2,view=view)
#                 plt.savefig('figures/coef_map_parcel/'+covariate+'_'+strategy+'_entropy_'+hemi+'_'+view, dpi=300, bbox_inches='tight')
#                 plt.close()


######################################## Coefficients from LME model ###############################################

for strategy in config['strategies']:
    df = pd.read_csv('data/results/logpower_stats_by_band_parcel_'+strategy+'_perm.csv')
    os.makedirs('figures/coef_map_parcel',exist_ok=True)
    for covariate in config['covariates']:
        if covariate=='SDI':
            continue
        if covariate=='PPL_mcg_L':
            vmin = -200
            vmax = 200
        elif covariate=='SDI':
            vmin = -0.02
            vmax = 0.02
        for band in config['frequency_bands']:
            df_reduced = df[(df['covariate']==covariate) & (df['uncontrolled'] == band)]
            avg_map = df_reduced.groupby('controlled')['coefcovariate'].mean()
            mask = df_reduced.groupby('controlled')['pval_perm'].apply(lambda x: (x <= p_value_threshold).all())
            # avg_map.iloc[df_reduced['pval']>p_value_threshold] = np.nan
            avg_map[~mask] = np.nan

            for hemi in ['lh', 'rh']:
                print(f'Visualizing {covariate} {strategy} {band} {hemi}')
                
                if hemi=='lh':
                    cortex_map = np.empty(masks['mask_lh'].shape)
                    cortex_map.fill(np.nan)
                    for roi_label, roi_number in zip(parcel_labels,unique_parcels):
                        if 'LH' in roi_label:
                            cortex_map[parcellation_extended['parcellation_lh_expanded']==roi_number] = avg_map[roi_label]
                    h = 'L'
                    h2 = 'left'
                elif hemi=='rh':
                    cortex_map = np.empty(masks['mask_rh'].shape)
                    cortex_map.fill(np.nan)
                    for roi_label, roi_number in zip(parcel_labels,unique_parcels):
                        if 'RH' in roi_label:
                            cortex_map[parcellation_extended['parcellation_rh_expanded']==roi_number] = avg_map[roi_label]
                    h = 'R'
                    h2 = 'right'
                fsLR_surface = surface.load_surf_mesh('data/external/fs_LR.32k.'+h+'.midthickness.surf.gii')
                for view in ['lateral','medial']:
                    plotting.plot_surf(fsLR_surface,cortex_map,symmetric_cmap=None,bg_on_data=False,colorbar=True, 
                                        cmap='bwr', vmin=vmin, vmax=vmax, darkness=None, hemi=h2,view=view)
                    plt.savefig('figures/coef_map_parcel/logpower_'+covariate+'_'+strategy+'_'+band+'_'+hemi+'_'+view, dpi=300, bbox_inches='tight')
                    plt.close()


# ######################################## Raw spectra, partial residuals in parcels ###############################################

# for strategy in config['strategies']:
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

#             for hemi in ['lh', 'rh']:
#                 print(f'Visualizing {strategy} {time_interval} {band} {hemi}')
                
#                 if hemi=='lh':
#                     cortex_map = np.empty(masks['mask_lh'].shape)
#                     cortex_map.fill(np.nan)
#                     for roi_label, roi_number in zip(parcel_labels,unique_parcels):
#                         if 'LH' in roi_label:
#                             cortex_map[parcellation_extended['parcellation_lh_expanded']==roi_number] = avg_map[roi_label]
#                     h = 'L'
#                     h2 = 'left'
#                 elif hemi=='rh':
#                     cortex_map = np.empty(masks['mask_rh'].shape)
#                     cortex_map.fill(np.nan)
#                     for roi_label, roi_number in zip(parcel_labels,unique_parcels):
#                         if 'RH' in roi_label:
#                             cortex_map[parcellation_extended['parcellation_rh_expanded']==roi_number] = avg_map[roi_label]
#                     h = 'R'
#                     h2 = 'right'
#                 fsLR_surface = surface.load_surf_mesh('data/external/fs_LR.32k.'+h+'.midthickness.surf.gii')
#                 plotting.plot_surf(fsLR_surface,cortex_map,symmetric_cmap=None,bg_on_data=False,colorbar=True, 
#                                     cmap='plasma', vmin=vmin, vmax=vmax, darkness=None, hemi=h2)
#                 plt.savefig('figures/spectrum_map_parcel/'+strategy+'_'+time_interval+'_'+band+'_'+hemi, dpi=300, bbox_inches='tight')
#                 plt.close()


# ######################################## Entropy, partial residuals in parcels ###############################################

# for strategy in config['strategies']:
#     df = pd.read_csv('data/results/entropy_by_parcel_'+strategy+'.csv')
#     for time_interval in config['time_intervals']:
#         df_reduced = df[(df['time_interval'] == time_interval)]
#         avg_map = df_reduced.groupby('roi')['partial_residuals'].mean()
        
#         vmin = 8
#         vmax = 8.5

#         for hemi in ['lh', 'rh']:
#             print(f'Visualizing {strategy} {time_interval} entropy {hemi}')
            
#             if hemi=='lh':
#                 cortex_map = np.empty(masks['mask_lh'].shape)
#                 cortex_map.fill(np.nan)
#                 for roi_label, roi_number in zip(parcel_labels,unique_parcels):
#                     if 'LH' in roi_label:
#                         cortex_map[parcellation_extended['parcellation_lh_expanded']==roi_number] = avg_map[roi_label]
#                 h = 'L'
#                 h2 = 'left'
#             elif hemi=='rh':
#                 cortex_map = np.empty(masks['mask_rh'].shape)
#                 cortex_map.fill(np.nan)
#                 for roi_label, roi_number in zip(parcel_labels,unique_parcels):
#                     if 'RH' in roi_label:
#                         cortex_map[parcellation_extended['parcellation_rh_expanded']==roi_number] = avg_map[roi_label]
#                 h = 'R'
#                 h2 = 'right'
#             fsLR_surface = surface.load_surf_mesh('data/external/fs_LR.32k.'+h+'.midthickness.surf.gii')
#             plotting.plot_surf(fsLR_surface,cortex_map,symmetric_cmap=None,bg_on_data=False,colorbar=True, 
#                                 cmap='plasma', vmin=vmin, vmax=vmax, darkness=None, hemi=h2)
#             plt.savefig('figures/spectrum_map_parcel/entropy_'+strategy+'_'+time_interval+'_'+hemi, dpi=300, bbox_inches='tight')
#             plt.close()

# ####################################### Raw spectra in voxels ###############################################

# df = pd.read_pickle('data/results/spectra_by_band_voxel.pkl')
# for strategy in config['strategies']:
#     for time_interval in config['time_intervals']:
#         for band in config['frequency_bands']:
#             df_reduced = df[(df['strategy'] == strategy) & (df['time_interval'] == time_interval) & (df['band'] == band)]
#             avg_map = np.stack(df_reduced['power']).mean((0,1))
#             # percentile vmin
#             # vmin = np.percentile(avg_map[avg_map > 0], 1)
#             # vmax = np.percentile(avg_map[avg_map > 0], 99)

#             if band == 'slow-3':
#                 vmin=2.7
#                 vmax=3.8
#             elif band == 'slow-4':
#                 vmin=2.4
#                 vmax=6.8
#             elif band == 'slow-5':
#                 vmin=2.3
#                 vmax=11

#             for hemi in ['lh', 'rh']:
#                 print(f'Visualizing {strategy} {time_interval} {band} {hemi}')
#                 mask = np.loadtxt('data/external/fsLR_32k_cortex-'+hemi+'_mask.txt', dtype=bool)
#                 cortex_map = np.zeros(mask.shape)
#                 if hemi=='lh':
#                     cortex_map[mask] = avg_map[:29696]
#                     h = 'L'
#                     h2 = 'left'
#                 elif hemi=='rh':
#                     cortex_map[mask] = avg_map[29696:59412]
#                     h = 'R'
#                     h2 = 'right'
#                 fsLR_surface = surface.load_surf_mesh('data/external/fs_LR.32k.'+h+'.midthickness.surf.gii')
#                 plotting.plot_surf(fsLR_surface,cortex_map,symmetric_cmap=None,bg_on_data=False,colorbar=True, 
#                                     cmap='black_red', vmin=vmin, vmax=vmax, darkness=None, hemi=h2, 
#                                     title=f'{strategy} {time_interval} {band} {hemi}')
#                 plt.savefig('figures/spectrum_map/'+strategy+'_'+time_interval+'_'+band+'_'+hemi, dpi=300, bbox_inches='tight')
#                 plt.close()
