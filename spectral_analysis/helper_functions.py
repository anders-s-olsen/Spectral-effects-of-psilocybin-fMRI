import numpy as np
import nibabel as nib
nib.imageglobals.logger.setLevel(40)
import matplotlib.pyplot as plt
import json
import pandas as pd
with open("config.json") as f:
    config = json.load(f)

import nilearn.surface as surface
import nilearn.plotting as plotting

def import_mask_and_parcellation(parcellation):
    """
    Import mask and parcellation data.
    
    Returns:
    parcel_labels (list): List of parcel labels.
    parcellation (np.array): Full parcellation array.
    masks (list): List containing left and right hemisphere masks.
    parcellation_expanded (list): List containing expanded left and right hemisphere parcellations
    """
    if parcellation == 'schaefer200':
        parcel_labels = np.loadtxt('data/external/Schaefer2018_200Parcels_7Networks_order_label.txt', dtype=str, usecols=0)
        parcel_labels = parcel_labels[0::2]
        parcellation_file = 'data/external/Schaefer2018_200Parcels_7Networks_order.dlabel.nii'
    elif parcellation == 'schaefertian232':
        parcel_labels = np.loadtxt('data/external/Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S2_label.txt', dtype=str, usecols=0)
        parcel_labels = parcel_labels[0::2]
        parcellation_file = 'data/external/Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S2.dlabel.nii'
    elif parcellation == 'schaefer1000':
        parcel_labels = np.loadtxt('data/external/Schaefer2018_1000Parcels_7Networks_order_label.txt', dtype=str, usecols=0)
        parcel_labels = parcel_labels[0::2]
        parcellation_file = 'data/external/Schaefer2018_1000Parcels_7Networks_order.dlabel.nii'
    # elif parcellation == 'raichle36':
    #     parcel_labels = np.loadtxt('data/external/Raichle2011_36Regions_collapsed_2mm.txt', dtype=str, usecols=0)
    #     parcellation_file = 'data/external/Raichle2011_36Regions_collapsed_2mm.nii'

    # ## Parcellation-wise power
    mask_lh = np.loadtxt('data/external/fsLR_32k_cortex-lh_mask.txt', dtype=bool)
    mask_rh = np.loadtxt('data/external/fsLR_32k_cortex-rh_mask.txt', dtype=bool)
    # parcel_labels[:32] = ['Subcort' + label for label in parcel_labels[:32]]
    parcellation_cortex = nib.load(parcellation_file).get_fdata()[0]
    parcellation_lh_expanded = parcellation_cortex[:mask_lh.shape[0]]
    parcellation_rh_expanded = parcellation_cortex[mask_lh.shape[0]:]
    parcellation_lh = parcellation_lh_expanded[mask_lh]
    parcellation_rh = parcellation_rh_expanded[mask_rh]
    parcellation = np.zeros((91282))
    parcellation[:parcellation_lh.shape[0]] = parcellation_lh
    parcellation[parcellation_lh.shape[0]:parcellation_lh.shape[0]+parcellation_rh.shape[0]] = parcellation_rh
    parcellation = parcellation.astype(int)
    # parcellation_lh_expanded = np.zeros(mask_lh.shape)
    # parcellation_rh_expanded = np.zeros(mask_rh.shape)
    # parcellation_lh_expanded[mask_lh] = parcellation_lh
    # parcellation_rh_expanded[mask_rh] = parcellation_rh

    unique_parcels = np.unique(parcellation)
    unique_parcels = unique_parcels[unique_parcels != 0]
    parcel_labels = parcel_labels[unique_parcels-1]  # -1 because parcels are 1-indexed in parcellation

    return parcel_labels, parcellation, {'mask_lh':mask_lh, 'mask_rh':mask_rh}, {'parcellation_lh_expanded':parcellation_lh_expanded, 'parcellation_rh_expanded':parcellation_rh_expanded}

def plot_surf(spatial_map, output_file, vmin, vmax):
    import hcp_utils as hcp
    parcel_labels, parcellation, masks, parcellation_extended = import_mask_and_parcellation(config['parcellation'])
    unique_parcels = np.unique(parcellation)
    unique_parcels = unique_parcels[unique_parcels != 0]  # remove background parcel
    # fsLR_surface_L = surface.load_surf_mesh('data/external/fs_LR.32k.L.midthickness.surf.gii')
    # fsLR_surface_R = surface.load_surf_mesh('data/external/fs_LR.32k.R.midthickness.surf.gii')
    # fsLR_surface = {'lh': fsLR_surface_L, 'rh': fsLR_surface_R}
    # if np.sum(spatial_map>0)> np.sum(spatial_map<0):
    #     spatial_map = -spatial_map  # flip sign for visualization consistency
    h2 = {'lh': 'left', 'rh': 'right'}
    h3 = {'lh': 'left', 'rh': 'right'}

    # if spatial maps is not a pd.Series with parcel labels as index, convert it to that    
    if not isinstance(spatial_map, pd.Series):
        spatial_map = pd.Series(spatial_map, index=parcel_labels)

    for hemi in ['lh', 'rh']:
        cortex_map = np.empty(masks['mask_'+hemi].shape)
        cortex_map.fill(np.nan)
        for roi_label, roi_number in zip(parcel_labels,unique_parcels):
            if hemi.upper() in roi_label:
                cortex_map[parcellation_extended['parcellation_'+hemi+'_expanded']==roi_number] = spatial_map[roi_label]
        
        for view in ['lateral','medial']:# medial
            plotting.plot_surf(hcp.mesh['midthickness_'+h3[hemi]],cortex_map,bg_map=hcp.mesh['sulc_'+h3[hemi]],
                               symmetric_cmap=None,bg_on_data=True,colorbar=True, 
                                cmap='bwr', vmin=vmin, vmax=vmax, alpha=1, hemi=h2[hemi],view=view,
                                output_file=output_file+'_'+hemi+'_'+view+'.png')
            # plotting.plot_img_on_surf

def pval_formatter(p):
    # p = p*7
    if p==0:
        return r'$p_{\text{FWER}} < 0.001$'
    else:
        return r'$p_{\text{FWER}}=$'+f'{p:.3f}'

def reorder_parcel_labels(parcel_labels):
    networks = config['networks']
    reordered_labels = []
    index = []
    for network in networks:
        for i,label in enumerate(parcel_labels):
            if network in label:
                reordered_labels.append(label)
                index.append(i)
    return reordered_labels, index

def plot_partial_residuals(df, target_variable, savename):
    # plot a covariate against partial residuals colored by scanner in one subplot, and covariate against target variable with a linear fit in another subplot, showing the correlation coefficient and p-value
    import seaborn as sns
    import scipy.stats as stats
    covariate = 'PPL_mcg/L'
    fig, axes = plt.subplots(3, 6, figsize=(20, 6))
    for i,y in enumerate([target_variable,'partial_residuals_onlyscanner','partial_residuals']):
        # scatter plot of covariate vs partial residuals colored by scanner
        sns.scatterplot(x=covariate, y=y, hue='scanner', ax=axes[i,0], data=df)
        # axes[i,0].set_title(f'{y} vs {covariate}')
        axes[i,0].set_xlabel(covariate)
        axes[i,0].set_ylabel(y)

        sns.scatterplot(x=covariate, y=y, hue='age', ax=axes[i,1], data=df)
        # axes[i,1].set_title(f'{y} vs {covariate}')
        axes[i,1].set_xlabel(covariate)
        axes[i,1].set_ylabel('')

        sns.scatterplot(x=covariate, y=y, hue='sex', ax=axes[i,2], data=df)
        # axes[i,2].set_title(f'{y} vs {covariate}')
        axes[i,2].set_xlabel(covariate)
        axes[i,2].set_ylabel('')

        sns.scatterplot(x='mean_fd', y=y, hue='scanner', ax=axes[i,3], data=df)
        # axes[i,3].set_title(f'{y} vs mean_fd')
        axes[i,3].set_xlabel('mean_fd')
        axes[i,3].set_ylabel('')

        sns.scatterplot(x='mean_std_dvars', y=y, hue='scanner', ax=axes[i,4], data=df)
        # axes[i,4].set_title(f'{y} vs mean_std_dvars')
        axes[i,4].set_xlabel('mean_std_dvars')
        axes[i,4].set_ylabel('')

        # scatter plot of covariate vs target variable with linear fit
        sns.regplot(x=covariate, y=y, ax=axes[i,5], scatter_kws={'s':10}, data=df)
        na_vals = df[covariate].isna() | df[y].isna()
        corr_coef, p_value = stats.pearsonr(df[covariate][~na_vals], df[y][~na_vals])
        axes[i,5].set_title(f'Correlation: {corr_coef:.2f}, p-value: {p_value:.3f}')
        axes[i,5].set_xlabel(covariate)
        axes[i,5].set_ylabel('')

    plt.tight_layout()
    plt.savefig('figures/partial_residuals/'+savename+'.png')
    plt.close()

# subcort_map = {'NAc-core':'accumbens_core',
#  'NAc-shell':'accumbens_shell',
#  'THA-DA':'thalamus_DA',
#  'THA-DP':'thalamus_DP',
#  'THA-VA':'thalamus_VA',
#  'THA-VP':'thalamus_VP',
#  'aCAU':'caudate_anterior',
#  'aGP':'pallidum_anterior',
#  'aHIP':'hippocampus_anterior',
#  'aPUT':'putamen_anterior',
#  'lAMY':'amygdala_lateral',
#  'mAMY':'amygdala_medial',
#  'pCAU':'caudate_posterior',
#  'pGP':'pallidum_posterior',
#  'pHIP':'hippocampus_posterior',
#  'pPUT':'putamen_posterior'}