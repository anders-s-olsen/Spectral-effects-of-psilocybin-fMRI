import numpy as np
import os
import pandas as pd
from scipy.linalg import eig
import matplotlib.pyplot as plt
import nitime.algorithms as tsa

def compute_ged(df,config, denoising_strategy):

    # compute GED for frequencies in specified range
    freq_min = config["ged_min_investigated_frequency"]
    freq_max = config["ged_max_investigated_frequency"]
    freq_step = config["ged_frequency_step_size"]
    frequencies = np.arange(freq_min, freq_max + freq_step, freq_step)
    frequencies = np.round(frequencies, decimals=3)

    entries = []
    
    for index, scan in df.iterrows():
        print(f"Processing scan: {scan.subject} {scan.run} with strategy {denoising_strategy}")
        denoised_dir = 'data/denoised/'+denoising_strategy+'/' + scan.subject + '/' + scan.session + '/func/'
        data = np.loadtxt(denoised_dir + os.path.basename(scan['preproc_filename_cifti']).replace(
            '.dtseries.nii', '_denoised_parcellated_schaefertian232.txt'))
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
        
        f,csd = tsa.multi_taper_csd(data.T, #assumes data is space x time
                                    Fs=fs,
                                    NW=None, # defaults to 4, which means 8 tapers
                                    BW=None, # defaults to None
                                    adaptive=False, # adaptive weighting of tapers, could be used (slow)
                                    low_bias=True, # Only use tapers with low bias, which is the default
                                    sides='default', #always onesided for non-complex-valued data
                                    NFFT=NFFT)
        # round frequencies to avoid floating point issues
        f = np.round(f, decimals=3)
        
        broadband_frequencies = (f >= config["ged_min_investigated_frequency"]) & (f <= config["ged_max_investigated_frequency"])
        
        cov_broadband = np.real(np.mean(csd[:,:,broadband_frequencies], axis=-1))
        cov_broadband *= p/np.trace(cov_broadband)

        # regularize broadband covariance
        cov_broadband = (1-config["broadband_regularization"])*cov_broadband + config["broadband_regularization"] * np.identity(cov_broadband.shape[0])

        cov_broadband_save = np.zeros((p,p))
        # impute nan rows and columns if present
        cov_broadband_save[:] = np.nan
        cov_broadband_save[np.ix_(~nan_columns, ~nan_columns)] = cov_broadband

        # cov_broadband_inv = np.linalg.inv(cov_broadband)

        # cov_broadband = compute_covariance_matrix(data, regularize=True, regularization_constant=config["broadband_regularization"])
        # leading_evecs_all = []
        cov_narrowband_all = []
        output_dir = 'data/ged_results/'+denoising_strategy+'/' + scan.subject + '/' + scan.session + '/func/'
        os.makedirs(output_dir, exist_ok=True)

        # filter data and compute covariance for each frequency
        for freq in frequencies:
            
            # compute narrowband covariance using CSD by applying a frequency domain gaussian filter
            freq_filter = np.exp(-0.5 * ((f - freq)/(freq_step))**2)
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
            plt.figure()
            plt.imshow(cov_narrowband, cmap='viridis', vmin=-1, vmax=1)
            # plt.ylim(-1,3)
            # title but with 2.float format
            # plt.title(f'Normalized narrowband covariance matrix at {freq:.2f} Hz for scan {scan.subject} {scan.run}')
            # plt.colorbar()
            plt.savefig(f'tmp_{freq:.2f}Hz_covariance.png', transparent=True)
            plt.close()

            # solve the GED problem
            evals, evecs = eig(cov_narrowband, cov_broadband)
            # evals, evecs = eig(cov_broadband_inv @ cov_narrowband)

            # sort eigenvalues and eigenvectors
            sorted_indices = np.argsort(evals)[::-1]
            evals = evals[sorted_indices]
            evecs = evecs[:, sorted_indices]
            num_evals_above_one = np.sum(evals.real > 1.0)

            # ensure evecs.T @ cov_broadband @ evecs = I
            evec_norms = np.sqrt(np.diag(evecs.T @ cov_broadband @ evecs))
            evecs = evecs / evec_norms[None,:]

            # impute nan columns with nans if present
            # evecs_imputed = np.empty((p, config['num_eigenvectors_to_keep']))
            evecs_imputed = np.empty((p, num_evals_above_one))
            evecs_imputed[:] = np.nan
            # evecs_imputed[~nan_columns,:] = evecs[:,:config['num_eigenvectors_to_keep']].real
            evecs_imputed[np.ix_(~nan_columns, np.arange(num_evals_above_one))] = evecs[:,:num_evals_above_one].real
            # leading_evecs_all.append(evecs_imputed)
            np.save(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_freq_'+str(freq)+'_ged_leading_eigenvectors.npy')), evecs_imputed)

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
        np.save(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_ged_narrowband_covariances.npy')), np.array(cov_narrowband_all))
        np.save(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_ged_broadband_covariance.npy')), cov_broadband_save)
        # save all entries to a dataframe
        df_ged = pd.DataFrame(entries)
        df_ged.to_csv('data/ged_results/ged_eigenvalues_'+denoising_strategy+'.csv', index=False)

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
        if denoising_strategy != 'acompcor':
            continue
        compute_ged(df, config, denoising_strategy)