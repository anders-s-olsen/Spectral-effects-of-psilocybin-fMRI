import numpy as np
import os
import pandas as pd
from scipy.linalg import eig, sqrtm, eigh
import matplotlib.pyplot as plt
import nitime.algorithms as tsa
from spectral_analysis.helper_functions import import_mask_and_parcellation, reorder_parcel_labels

def compute_ged(df,config, denoising_strategy):

    # compute GED for frequencies in specified range
    freq_min = config["ged_min_investigated_frequency"]
    freq_max = config["ged_max_investigated_frequency"]
    freq_step = config["ged_frequency_step_size"]
    frequencies = np.arange(freq_min, freq_max + freq_step, freq_step)
    frequencies = np.round(frequencies, decimals=2)

    entries = []

    labels = import_mask_and_parcellation(config['parcellation'])[0]
    labels_reordered,reorder_index = reorder_parcel_labels(labels)
    
    for index, scan in df.iterrows():
        print(f"Processing scan: {scan.subject} {scan.run} with strategy {denoising_strategy}")
        denoised_dir = 'data/denoised/'+denoising_strategy+'/' + scan.subject + '/' + scan.session + '/func/'
        data = np.loadtxt(denoised_dir + os.path.basename(scan['preproc_filename_cifti']).replace(
            '.dtseries.nii', '_denoised_parcellated_'+config['parcellation']+'.txt'))
        n, p = data.shape

        # find nan columns and remove them
        nan_columns = np.isnan(data).any(axis=0)
        if np.sum(nan_columns) > 0:
            print(f"Warning: Found {np.sum(nan_columns)} parcels with NaN values, removing them for GED computation")
            data = data[:, ~nan_columns]

        # compute broadband covariance
        data -= np.mean(data, axis=0) # should be redundant
        fs = 1/scan.tr
        
        # set NFFT based on scanner such that frequencies are spaced apart by 0.01Hz
        if scan.scanner == 'MR001':
            NFFT = 1000
        elif scan.scanner == 'MR45':
            NFFT = 400
        
        frequencies1,csd = tsa.multi_taper_csd(data.T, #assumes data is space x time
                                    Fs=fs,
                                    NW=4, # defaults to 4, which means 7 tapers
                                    BW=None, # defaults to None
                                    adaptive=False, # adaptive weighting of tapers, could be used (slow)
                                    low_bias=True, # Only use tapers with low bias, which is the default
                                    sides='default', #always onesided for non-complex-valued data
                                    NFFT=NFFT)
        # round frequencies to avoid floating point issues
        frequencies1 = np.round(frequencies1, decimals=5)
        
        broadband_frequencies = (frequencies1 >= config["ged_min_investigated_frequency"]) & (frequencies1 <= config["ged_max_investigated_frequency"])
        
        cov_broadband = np.real(np.mean(csd[:,:,broadband_frequencies], axis=-1))
        cov_broadband *= p/np.trace(cov_broadband)

        # regularize broadband covariance
        cov_broadband = (1-config["broadband_regularization"])*cov_broadband + config["broadband_regularization"] * np.identity(cov_broadband.shape[0])

        cov_broadband_save = np.zeros((p,p))
        # impute nan rows and columns if present
        cov_broadband_save[:] = np.nan
        cov_broadband_save[np.ix_(~nan_columns, ~nan_columns)] = cov_broadband
        # plt.figure()
        # cov_broadband_reordered = cov_broadband_save[reorder_index,:][:,reorder_index]
        # plt.imshow(cov_broadband_reordered, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        # plt.xticks([])
        # plt.yticks([])
        # # plt.savefig(f'figures/narrowband_csd/tmp_broadband_covariance.png', transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.savefig('tmp.png')
        # plt.close()

        # cov_broadband_inv = np.linalg.inv(cov_broadband)

        # cov_broadband = compute_covariance_matrix(data, regularize=True, regularization_constant=config["broadband_regularization"])
        # leading_evecs_all = []
        cov_narrowband_all = []
        output_dir = 'data/ged_results/'+denoising_strategy+'/' + scan.subject + '/' + scan.session + '/func/'
        os.makedirs(output_dir, exist_ok=True)

        eigenvalues_across_frequency = np.zeros((len(frequencies), config["num_eigenvectors_to_keep"]))

        # filter data and compute covariance for each frequency
        for f,freq in enumerate(frequencies):
            freq_str = f"{freq:.2f}"
            print(f"Computing GED for frequency {freq_str}Hz",end='\r')
            
            # compute narrowband covariance using CSD by applying a frequency domain gaussian filter
            freq_filter = np.exp(-0.5 * ((frequencies1 - freq)/(freq_step))**2)
            csd_filtered = csd * freq_filter[None,None,:]
            cov_narrowband = np.real(np.mean(csd_filtered, axis=-1))

            # normalize
            cov_narrowband *= p/np.trace(cov_narrowband)
            cov_narrowband_save = np.zeros((p,p))
            # impute nan rows and columns if present
            cov_narrowband_save[:] = np.nan
            # cov_narrowband_save[~nan_columns,:][:,~nan_columns] = cov_narrowband
            cov_narrowband_save[np.ix_(~nan_columns, ~nan_columns)] = cov_narrowband
            cov_narrowband_all.append(cov_narrowband_save)
            # plt.figure()
            # cov_narrowband_reordered = cov_narrowband[reorder_index,:][:,reorder_index]
            # plt.imshow(cov_narrowband_reordered, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig(f'figures/narrowband_csd/tmp_{freq_str}Hz_covariance.png', transparent=True, bbox_inches='tight', pad_inches=0)
            # plt.close()

            # solve the GED problem
            evals, evecs = eigh(cov_narrowband, cov_broadband)
            # evals, evecs = eig(cov_broadband_inv @ cov_narrowband)

            if np.any(evals.imag > 1e-6) or np.any(np.isinf(evals)) or np.any(np.isnan(evals)):
                print(f"Warning: Found {np.sum(evals.imag > 1e-6)} complex, {np.sum(np.isinf(evals))} infinite, and {np.sum(np.isnan(evals))} NaN eigenvalues at frequency {freq}Hz for scan {scan.subject} {scan.run} with strategy {denoising_strategy}")
                # continue

            # sort eigenvalues and eigenvectors
            sorted_indices = np.argsort(evals)[::-1]
            evals = evals[sorted_indices]
            evecs = evecs[:, sorted_indices]
            # num_evals_above_one = np.sum(evals.real > 1.0)
            num_evals_above_one = config['num_eigenvectors_to_keep']
            eigenvalues_across_frequency[f,:] = evals[:config['num_eigenvectors_to_keep']].real

            # ensure evecs.T @ cov_broadband @ evecs = I
            # evec_norms = np.sqrt(np.diag(evecs.T @ cov_broadband @ evecs))
            # evecs = evecs / evec_norms[None,:]

            # impute nan columns with nans if present
            # evecs_imputed = np.empty((p, config['num_eigenvectors_to_keep']))
            evecs_imputed = np.empty((p, num_evals_above_one))
            evecs_imputed[:] = np.nan
            evecs_imputed[np.ix_(~nan_columns, np.arange(num_evals_above_one))] = evecs[:,:num_evals_above_one].real
            evecs_imputed_rescaled = np.empty((p, num_evals_above_one))
            evecs_imputed_rescaled[:] = np.nan
            evecs_imputed_rescaled[np.ix_(~nan_columns, np.arange(num_evals_above_one))] = cov_broadband @ evecs[:,:num_evals_above_one]

            # print('Imaginary part of leading eigenvector (should be close to 0): ', np.linalg.norm(evecs[:,:num_evals_above_one].imag))
            # leading_evecs_all.append(evecs_imputed)
            # evecs_imputed_rescaled = sqrtm(cov_broadband_save) @ evecs_imputed @ np.diag(np.sqrt(evals[:num_evals_above_one].real))
            np.save(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_freq_'+freq_str+'_ged_leading_eigenvectors.npy')), evecs_imputed)
            np.save(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_freq_'+freq_str+'_ged_leading_eigenvectors_rescaled.npy')), evecs_imputed_rescaled)
            np.save(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_freq_'+freq_str+'_ged_narrowband_covariance.npy')), cov_narrowband_save)

            for eval_idx in range(num_evals_above_one):
            # for eval_idx in range(config['num_eigenvectors_to_keep']):
                entries.append(scan.to_dict() | { 
                    'denoising_strategy': denoising_strategy,
                    'frequency': freq,
                    'eigenvector_index': eval_idx + 1,
                    'eigenvalues': evals[eval_idx].real,
                    'variance_explained': (evals[eval_idx].real / np.sum(evals.real)),
                    'variance_explained_of_evals_above_one': (evals[eval_idx].real / np.sum(evals.real[evals.real > 1.0])),
                })

        # save results for this scan
        # np.savetxt(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_ged_eigenvalues.txt')), np.vstack(evals_all))
        # np.save(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_ged_leading_eigenvectors.npy')), np.array(leading_evecs_all))
        # np.save(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_ged_narrowband_covariances.npy')), np.array(cov_narrowband_all))
        np.save(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_ged_broadband_covariance.npy')), cov_broadband_save)
        # save all entries to a dataframe
        df_ged = pd.DataFrame(entries)
        df_ged.to_csv('data/ged_results/ged_eigenvalues_'+denoising_strategy+'.csv', index=False)
        np.savetxt(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_ged_eigenvalues.txt')), eigenvalues_across_frequency)

if __name__ == "__main__":
    import json

    with open("config.json") as f:
        config = json.load(f)

    # Load the DataFrame containing scan information
    df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
    df = df[df['task']==config["task"]]
    df = df[df['include_scan_coil_numvols']]
    df = df[df['include_manual_qc']]
    df = df[df['ratio_outliers_fd0.5_std_dvars1000'] < config["max_ratio_outliers_fd0.5_std_dvars1000"]]
    df = df[df['max_fd'] < config["scan_max_fd_threshold"]]
    strategies = config["strategies"]
    
    # Compute spectra
    for denoising_strategy in strategies:
        if denoising_strategy not in ['9p']:
            continue
        compute_ged(df, config, denoising_strategy)