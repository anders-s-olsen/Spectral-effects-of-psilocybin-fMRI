import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
def import_mask_and_parcellation():
    """
    Import mask and parcellation data.
    
    Returns:
    parcel_labels (list): List of parcel labels.
    parcellation (np.array): Full parcellation array.
    masks (list): List containing left and right hemisphere masks.
    parcellation_expanded (list): List containing expanded left and right hemisphere parcellations
    """
    # ## Parcellation-wise power
    mask_lh = np.loadtxt('data/external/fsLR_32k_cortex-lh_mask.txt', dtype=bool)
    mask_rh = np.loadtxt('data/external/fsLR_32k_cortex-rh_mask.txt', dtype=bool)
    parcel_labels = np.loadtxt('data/external/Schaefer2018_200Parcels_7Networks_order_label.txt', dtype=str, usecols=0)
    parcel_labels = parcel_labels[0::2]
    # parcel_labels[:32] = ['Subcort' + label for label in parcel_labels[:32]]
    parcellation_file = 'data/external/Schaefer2018_200Parcels_7Networks_order.dlabel.nii'
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

    return parcel_labels, parcellation, {'mask_lh':mask_lh, 'mask_rh':mask_rh}, {'parcellation_lh_expanded':parcellation_lh_expanded, 'parcellation_rh_expanded':parcellation_rh_expanded}

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