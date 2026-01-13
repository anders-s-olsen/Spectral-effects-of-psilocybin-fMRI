# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
import json
from scipy.stats import chi2
from scipy.linalg import sqrtm
with open('config.json', 'r') as f:
    config = json.load(f)

# %%
# for only acompcor, load the eigenvectors and display the average correlation structure over frequencies (average over subejcts)
import os
strategy = 'acompcor'
print(f'Analyzing denoising strategy: {strategy}')
df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_task-rest_PPLSDI.csv')
df = df[df['task']==config["task"]]
df = df[df['include_scan_coil_numvols']]
df = df[df['include_manual_qc']]
df = df[df['ratio_outliers_fd0.5_std_dvars1000'] < config["max_ratio_outliers_fd0.5_std_dvars1000"]]
df = df[df['max_fd'] < config["scan_max_fd_threshold"]]
eigenvectors_list = []
cov_broadband_list = []
cov_narrowband_list = []
ppl_list = []
for index, scan in df.iterrows():
    output_dir = 'data/ged_results/'+strategy+'/' + scan.subject + '/' + scan.session + '/func/'
    eigenvectors = np.load(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_ged_leading_eigenvectors.npy')))
    eigenvectors_list.append(eigenvectors)
    ppl_list.append(scan['PPL_mcg/L'])  
    cov_broadband_list.append(np.load(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_ged_broadband_covariance.npy'))))
    cov_narrowband_list.append(np.load(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_ged_narrowband_covariances.npy'))))
eigenvectors_array = np.array(eigenvectors_list)  # shape: (num_scans, num_frequencies, num_vertices, num_eigenvectors)
cov_broadband_array = np.array(cov_broadband_list)  # shape: (num_scans, num_vertices, num_vertices)
cov_narrowband_array = np.array(cov_narrowband_list)  # shape: (num_scans, num_frequencies, num_vertices, num_vertices)

cov_broadband_array_sqrtm = np.empty_like(cov_broadband_array)
orthonormal_eigenvectors_array = np.empty_like(eigenvectors_array)
for i in range(eigenvectors_array.shape[0]):
    nan_idx = np.isnan(cov_broadband_array[i]).all(axis=1)
    cov_broadband_array_nonan = cov_broadband_array[i][np.ix_(~nan_idx, ~nan_idx)]
    cov_broadband_array_sqrtm_i = sqrtm(cov_broadband_array_nonan)
    orthonormal_eigenvectors_array_i = cov_broadband_array_sqrtm_i[None] @ eigenvectors_array[i][:,~nan_idx,:]
    for ev in range(orthonormal_eigenvectors_array_i.shape[-1]):
        if np.sum(orthonormal_eigenvectors_array_i[:,ev]>0)> np.sum(orthonormal_eigenvectors_array_i[:,ev]<0):
            orthonormal_eigenvectors_array_i[:,ev] = -orthonormal_eigenvectors_array_i[:,ev]  # flip sign for consistency
    cov_broadband_array_sqrtm[i][np.ix_(~nan_idx, ~nan_idx)] = cov_broadband_array_sqrtm_i
    orthonormal_eigenvectors_array[i][:,~nan_idx,:] = orthonormal_eigenvectors_array_i
spatial_maps_array = cov_broadband_array[:,None,:,:] @ eigenvectors_array

freq_min = config["ged_min_investigated_frequency"]
freq_max = config["ged_max_investigated_frequency"]
freq_step = config["ged_frequency_step_size"]
frequencies = np.arange(freq_min, freq_max + freq_step, freq_step)
frequencies = np.round(frequencies, decimals=3) 

# %%
# from spectral_analysis.helper_functions import import_mask_and_parcellation
# import nilearn.surface as surface
# import nilearn.plotting as plotting
# from time import time
# os.makedirs('figures/ged_subject_specific_maps/', exist_ok=True)
# parcel_labels, parcellation, masks, parcellation_extended = import_mask_and_parcellation()
# unique_parcels = np.unique(parcellation)
# unique_parcels = unique_parcels[unique_parcels != 0]  # remove background parcel
# fsLR_surface_L = surface.load_surf_mesh('data/external/fs_LR.32k.L.midthickness.surf.gii')
# fsLR_surface_R = surface.load_surf_mesh('data/external/fs_LR.32k.R.midthickness.surf.gii')
# fsLR_surface = {'lh': fsLR_surface_L, 'rh': fsLR_surface_R}
# for scan in range(eigenvectors_array.shape[0]):
#     for freq in range(eigenvectors_array.shape[1]):
#         for eigenvector in range(eigenvectors_array.shape[-1]): 
#             spatial_map = spatial_maps_array[scan,freq,:,eigenvector]
#             if np.sum(spatial_map>0)> np.sum(spatial_map<0):
#                 spatial_map = -spatial_map  # flip sign for visualization consistency
#             for hemi in ['lh', 'rh']:
#                 t1 = time()
#                 if hemi=='lh':
#                     h2 = 'left'
#                 elif hemi=='rh':
#                     h2 = 'right'
#                 cortex_map = np.empty(masks['mask_'+hemi].shape)
#                 cortex_map.fill(np.nan)
#                 for roi_label, roi_number in zip(parcel_labels,unique_parcels):
#                     if hemi.upper() in roi_label:
#                         cortex_map[parcellation_extended['parcellation_'+hemi+'_expanded']==roi_number] = spatial_map[roi_number-1]

#                 t2 = time()
#                 print(f'Plotting scan {scan+1}/{spatial_maps_array.shape[0]}, freq {frequencies[freq]:.3f} Hz, eigenvector {eigenvector+1}, hemi {hemi}, time taken: {t2-t1:.2f} seconds')
#                 for view in ['lateral']:# medial
#                     plotting.plot_surf(fsLR_surface[hemi],cortex_map,symmetric_cmap=None,bg_on_data=False,colorbar=True, 
#                                         cmap='bwr', vmin=-0.2, vmax=0.2, darkness=None, hemi=h2,view=view,
#                                         output_file='figures/ged_subject_specific_maps/'+strategy+'_'+str(scan)+'_'+str(frequencies[freq])+'_'+str(eigenvector)+'_'+hemi+'_'+view+'.png')
#                     t3 = time()
#                     print(f'  Plot generated in {t3-t2:.2f} seconds')
#                     plt.close()

# %%
def generalized_procrustes_subspaces(bases, tol=1e-9, max_iter=1000):
    """
    Generalized Procrustes Analysis (GPA) for subspaces.

    Each subspace is represented by an orthonormal basis matrix of size (P, Q).
    The algorithm aligns all subspaces by optimal orthogonal transforms
    to minimize the total Frobenius distance.

    Parameters
    ----------
    bases : np.ndarray
        Array of shape (N, P, Q)
        Each element bases[i] is an orthonormal basis of subspace i.
    tol : float
        Convergence tolerance (default 1e-9)
    max_iter : int
        Maximum number of iterations (default 1000)

    Returns
    -------
    aligned_bases : np.ndarray
        Array of shape (N, P, Q), aligned orthonormal bases.
    mean_basis : np.ndarray
        Orthonormal mean basis of size (P, Q).
    """
    N, P, Q = bases.shape

    # Ensure each basis is orthonormal
    # bases = np.array([np.linalg.qr(b)[0] for b in bases])

    # Initialize mean basis
    mean_basis = bases[0]
    delta_all = []

    for _ in range(max_iter):
        aligned = np.zeros_like(bases)
        distance = np.zeros(N)

        # Align each subspace to current mean
        for i, B in enumerate(bases):
            U, _, Vt = np.linalg.svd(mean_basis.T @ B)
            R = Vt.T @ U.T
            aligned[i] = B @ R
            distance[i] = np.linalg.norm(mean_basis - aligned[i])**2

        # Compute new mean subspace
        M = np.zeros((P, P))
        for A in aligned:
            M += A@A.T
        # Re-orthogonalize mean
        # new_mean, _ = np.linalg.qr(M)
        new_mean = np.linalg.eig(M)[1][:, :Q]

        # Check convergence (principal angle change)
        delta = np.linalg.norm(mean_basis - new_mean)
        mean_basis = new_mean
        bases = aligned
        delta_all.append(delta)
        if delta < tol:
            break

    return aligned, mean_basis, distance, delta_all



# %%
os.makedirs('figures/ged_mean_basis_maps/', exist_ok=True)
from spectral_analysis.helper_functions import import_mask_and_parcellation
import nilearn.surface as surface
import nilearn.plotting as plotting
parcel_labels, parcellation, masks, parcellation_extended = import_mask_and_parcellation()
unique_parcels = np.unique(parcellation)
unique_parcels = unique_parcels[unique_parcels != 0]  # remove background parcel
fsLR_surface_L = surface.load_surf_mesh('data/external/fs_LR.32k.L.midthickness.surf.gii')
fsLR_surface_R = surface.load_surf_mesh('data/external/fs_LR.32k.R.midthickness.surf.gii')
fsLR_surface = {'lh': fsLR_surface_L, 'rh': fsLR_surface_R}
for freq in range(eigenvectors_array.shape[1]):
    aligned, mean_basis, distance, delta = generalized_procrustes_subspaces(orthonormal_eigenvectors_array[:,freq], tol=1e-9, max_iter=500)
    for eigenvector in [0]: #  range(mean_basis.shape[-1])
        if np.sum(mean_basis[:,eigenvector]>0) > np.sum(mean_basis[:,eigenvector]<0):
            mean_basis[:,eigenvector] = -mean_basis[:,eigenvector]  # flip sign for consistency
        for hemi in ['lh', 'rh']:
            if hemi=='lh':
                h2 = 'left'
            elif hemi=='rh':
                h2 = 'right'
            cortex_map = np.empty(masks['mask_'+hemi].shape)
            cortex_map.fill(np.nan)
            for roi_label, roi_number in zip(parcel_labels,unique_parcels):
                if hemi.upper() in roi_label:
                    cortex_map[parcellation_extended['parcellation_'+hemi+'_expanded']==roi_number] = mean_basis[roi_number-1,eigenvector]
            
            for view in ['lateral']:# medial
                plotting.plot_surf(fsLR_surface[hemi],cortex_map,symmetric_cmap=None,bg_on_data=False,colorbar=True, 
                                    cmap='bwr', vmin=-0.1, vmax=0.1, darkness=None, hemi=h2,view=view,
                                    output_file='figures/ged_mean_basis_maps/'+strategy+'_'+str(frequencies[freq])+'_'+str(eigenvector)+'_'+hemi+'_'+view+'.png')

# %%
from spectral_analysis.helper_functions import import_mask_and_parcellation
import nilearn.surface as surface
import nilearn.plotting as plotting
parcel_labels, parcellation, masks, parcellation_extended = import_mask_and_parcellation()
unique_parcels = np.unique(parcellation)
unique_parcels = unique_parcels[unique_parcels != 0]  # remove background parcel
fsLR_surface_L = surface.load_surf_mesh('data/external/fs_LR.32k.L.midthickness.surf.gii')
fsLR_surface_R = surface.load_surf_mesh('data/external/fs_LR.32k.R.midthickness.surf.gii')
fsLR_surface = {'L': fsLR_surface_L, 'R': fsLR_surface_R}
for freq in range(eigenvectors_array.shape[1]):
    aligned, mean_basis, distance, delta = generalized_procrustes_subspaces(eigenvectors_array[:, freq,:,:], tol=1e-9, max_iter=500)
    eigenvectors_freq = aligned
    # eigenvectors_freq = eigenvectors_array[:, freq,:,:]
    spatial_maps = cov_broadband_array @ eigenvectors_freq  # shape: (num_scans, num_vertices, num_eigenvectors)
    # sort the sign of each spatial map to have consistent polarity across subjects
    # for sub in range(spatial_maps.shape[0]):
    #     for ev in range(spatial_maps.shape[-1]):
    #         if spatial_maps[sub, :, ev].mean() > 0:
    #             spatial_maps[sub, :, ev] *= -1
    
    spatial_maps_mean = np.mean(spatial_maps, axis=0)  # shape: (num_vertices, num_eigenvectors)
    for eigenvector in [0]: #  range(mean_basis.shape[-1])
        for hemi in ['lh', 'rh']:
            if hemi=='lh':
                cortex_map = np.empty(masks['mask_lh'].shape)
                cortex_map.fill(np.nan)
                for roi_label, roi_number in zip(parcel_labels,unique_parcels):
                    if 'LH' in roi_label:
                        cortex_map[parcellation_extended['parcellation_lh_expanded']==roi_number] = spatial_maps_mean[roi_number-1,eigenvector]
                h = 'L'
                h2 = 'left'
            elif hemi=='rh':
                cortex_map = np.empty(masks['mask_rh'].shape)
                cortex_map.fill(np.nan)
                for roi_label, roi_number in zip(parcel_labels,unique_parcels):
                    if 'RH' in roi_label:
                        cortex_map[parcellation_extended['parcellation_rh_expanded']==roi_number] = spatial_maps_mean[roi_number-1,eigenvector]
                h = 'R'
                h2 = 'right'
            for view in ['lateral']:# medial
                plotting.plot_surf(fsLR_surface[h],cortex_map,symmetric_cmap=None,bg_on_data=False,colorbar=True, 
                                    cmap='bwr', vmin=-0.05, vmax=0.05, darkness=None, hemi=h2,view=view)
                os.makedirs('figures/ged_spatial_maps/', exist_ok=True)
                plt.savefig('figures/ged_spatial_maps/'+strategy+'_'+str(frequencies[freq])+'_'+str(eigenvector)+'_'+hemi+'_'+view+'.png', dpi=300, bbox_inches='tight')
                plt.close()


