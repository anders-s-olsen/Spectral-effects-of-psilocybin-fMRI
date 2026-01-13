import pandas as pd
import os 
import nibabel as nib
import numpy as np
from glob import glob
from spectral_analysis.helper_functions import import_mask_and_parcellation
from scipy.stats import kurtosis
def parcellate_timeseries(df, denoising_strategy, config):
    """
    Parcellate the time series of the scans in the DataFrame df.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing scan information.
    despiking (bool): If True, apply despiking to the time series.
    
    Returns:
    None
    """
    # load parcellation
    parcel_labels, parcellation, masks, parcellation_extended = import_mask_and_parcellation()
    unique_parcels = np.unique(parcellation)
    unique_parcels = unique_parcels[unique_parcels != 0]  # remove background parcel
    # find denoising directories
    denoised_top_dir = 'data/denoised/'+denoising_strategy
    preproc_top_dir = 'data/preprocessed/'

    for index, scan in df.iterrows():
        print(f"Processing scan: {scan.subject} {scan.session} {scan.task} {scan.run}")

        denoised_dir = denoised_top_dir + '/' + scan.subject + '/' + scan.session + '/func/'
        preproc_dir = preproc_top_dir + '/' + scan.subject + '/' + scan.session + '/func/'
        try:
            denoised_files = glob(os.path.join(denoised_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_denoised*.dtseries.nii')))
            tsnr = nib.load(os.path.join(preproc_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_tsnr.dscalar.nii'))).get_fdata()
        except:
            print(f"Scan {scan['preproc_filename_cifti']} does not exist, skipping")
            continue
        for denoised_file in denoised_files:
            if 'filtered' in denoised_file:
                continue
            # if os.path.exists(denoised_file.replace('.dtseries.nii', '_parcellated_schaefertian232.txt')):
            #     print(f"Parcellated time series already exists for scan {scan['preproc_filename_cifti']}, skipping")
            #     continue

            # Load the scan
            img = nib.load(denoised_file)
            data = img.get_fdata()

            # dead voxels set to nan
            data[:,(data**2).sum(axis=0) == 0] = np.nan

            parcel_time_series = np.zeros((data.shape[0], len(unique_parcels)))

            for i,parcel in enumerate(unique_parcels):
                # Create a mask for the current parcel
                mask = (parcellation == parcel) & (tsnr[0,:] > config['min_tsnr'])  # only include voxels with tSNR > 0
                tmp = data[:, mask]
                tmp[:,np.any(tmp>config['voxel_max_psc_threshold'],axis=0)] = np.nan  # extreme values set to nan
                tmp[:,kurtosis(tmp,axis=0)>config['voxel_max_kurtosis_threshold']] = np.nan  # extreme kurtosis set to nan
                
                if np.sum(mask) == 0:
                    raise_error = True
                elif np.isnan(tmp).all():
                    raise_error = True
                elif np.any(np.sum(tmp**2,0)==0):
                    raise_error = True
                elif np.any(np.std(tmp,axis=0)==0):
                    raise_error = True
                elif tmp.shape[1]<5:
                    raise_error = True
                else:
                    raise_error = False
                if raise_error:
                    if scan['ratio_outliers_fd0.5_std_dvars1000'] < config["max_ratio_outliers_fd0.5_std_dvars1000"] and scan['max_fd'] < config["scan_max_fd_threshold"]:
                        print(f"Error: Parcel {parcel} has issues in scan {scan['preproc_filename_cifti']}, but scan is included in analysis.")
                        parcel_time_series[:,i] = np.nan
                        continue
                    else:
                        print(f"Warning: Parcel {parcel} has issues in scan {scan['preproc_filename_cifti']}, setting to NaN")
                        parcel_time_series[:,i] = np.nan
                        continue
                
                # Extract the time series for the current parcel
                parcel_time_series[:,i] = np.nanmean(tmp,axis=1)

            # Save the time series to a file
            output_file = denoised_file.replace('.dtseries.nii', '_parcellated_schaefertian232.txt')
            np.savetxt(output_file, parcel_time_series)
        
if __name__ == "__main__":

    import json
    with open("config.json") as f:
        config = json.load(f)

    strategies = config["strategies"]

    df = pd.read_csv('data/func_scans_table_outliers.csv')
    df = df[df['session']==config["session"]]
    df = df[df['task']==config["task"]]
    df = df[df['include_scan_coil_numvols']]
    df = df[df['include_manual_qc']]

    for denoising_strategy in strategies:
        print(f"Parcellating time series for strategy: {denoising_strategy}")        
        # Parcellate the time series
        parcellate_timeseries(df, denoising_strategy, config)