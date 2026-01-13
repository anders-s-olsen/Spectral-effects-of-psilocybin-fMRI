import pandas as pd
import os 
import nibabel as nib
import numpy as np
from spectral_analysis.helper_functions import import_mask_and_parcellation

def parcellate_spectrum(df, denoising_strategy, config, downsample_mr001=False):
    """
    Parcellate the time series of the scans in the DataFrame df.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing scan information.
    despiking (bool): If True, apply despiking to the time series.
    
    Returns:
    None
    """
    # load parcellation
    _, parcellation, _, _ = import_mask_and_parcellation()
    unique_parcels = np.unique(parcellation)
    unique_parcels = unique_parcels[unique_parcels != 0]  # remove background parcel

    # find denoising directories
    spectrum_top_dir = 'data/spectra/'+denoising_strategy

    for index, scan in df.iterrows():

        if downsample_mr001 and scan.scanner == 'MR001':
            add_downsampled_label = '_downsampled'
        else:
            add_downsampled_label = ''

        spectrum_dir = spectrum_top_dir + '/' + scan.subject + '/' + scan.session + '/func/'
        preproc_dir = 'data/preprocessed/' + scan.subject + '/' + scan.session + '/func/'

        for filetype in ['_mtspectra.dtseries.nii','_mtspectra_entropy.dscalar.nii']:#,,
            try:
                spectrum_file = os.path.join(spectrum_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', add_downsampled_label+filetype))
                if 'dtseries' in filetype:
                    if os.path.exists(spectrum_file.replace('.dtseries.nii', '_parcellated_schaefertian232.txt')):
                        print(f"Parcellated spectrum already exists for scan {os.path.basename(scan['preproc_filename_cifti'])}, skipping")
                        continue
                elif 'dscalar' in filetype:
                    if os.path.exists(spectrum_file.replace('.dscalar.nii', '_parcellated_schaefertian232.txt')):
                        print(f"Parcellated spectrum already exists for scan {os.path.basename(scan['preproc_filename_cifti'])}, skipping")
                        continue
                img = nib.load(spectrum_file)
                tsnr = nib.load(os.path.join(preproc_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_tsnr.dscalar.nii'))).get_fdata()
            except:
                print(f"Scan {os.path.basename(scan['preproc_filename_cifti'])} does not exist, skipping")
                continue
            print(f"Processing scan: {scan.subject} {scan.session} {scan.task} {scan.run}")

            # Load the scan
            data = img.get_fdata()
            if 'mtspectra' in filetype:
                data = 10*np.log10(data)  # convert to dB

            # dead voxels set to nan
            data[:,(data**2).sum(axis=0) == 0] = np.nan
            # inf to nan
            data[np.isinf(data)] = np.nan

            parcel_spectrum = np.zeros((data.shape[0], len(unique_parcels)))

            for i,parcel in enumerate(unique_parcels):
                # Create a mask for the current parcel
                mask = (parcellation == parcel) & (tsnr[0,:] > config['min_tsnr'])  # only include voxels with tSNR > 0
                if np.sum(mask) == 0:
                    raise_error = True
                    # raise ValueError(f"Warning: Parcel {parcel} not found in parcellation, skipping")
                elif np.isnan(data[:, mask]).all():
                    raise_error = True
                    # raise ValueError(f"Warning: All values in parcel {parcel} are NaN, skipping")
                elif np.any(np.sum(data[:,mask]**2,0)==0):
                    raise_error = True
                    # raise ValueError(f"Parcel {parcel} has no data in scan {scan['preproc_filename_cifti']}")
                else:
                    raise_error = False
                if raise_error:
                    if scan['ratio_outliers_fd0.5_std_dvars1000'] < config["max_ratio_outliers_fd0.5_std_dvars1000"] and scan['max_fd'] < config["scan_max_fd_threshold"]:
                        print(f"Error: Parcel {parcel} has issues in scan {scan['preproc_filename_cifti']}, but scan is included in analysis.")
                        parcel_spectrum[:,i] = np.nan
                        continue
                    else:
                        print(f"Warning: Parcel {parcel} has issues in scan {scan['preproc_filename_cifti']}, setting to NaN")
                        parcel_spectrum[:,i] = np.nan
                        continue
                
                # Extract the time series for the current parcel
                parcel_spectrum[:,i] = np.nanmean(data[:, mask],axis=-1)

            # Save the time series to a file
            if 'dtseries' in filetype:
                output_file = spectrum_file.replace('.dtseries.nii', '_parcellated_schaefertian232.txt')
            elif 'dscalar' in filetype:
                output_file = spectrum_file.replace('.dscalar.nii', '_parcellated_schaefertian232.txt')
            np.savetxt(output_file, parcel_spectrum)

if __name__ == "__main__":
    import json

    with open("config.json") as f:
        config = json.load(f)

    strategies = config["strategies"]

    # Load the DataFrame containing scan information
    df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
    df = df[df['task']==config["task"]]
    df = df[df['include_scan_coil_numvols']]
    df = df[df['include_manual_qc']]

    downsample_mr001 = False

    for denoising_strategy in strategies:
        print(f"Parcellating spectra for denoising strategy: {denoising_strategy}")
        parcellate_spectrum(df, denoising_strategy, config=config, downsample_mr001=downsample_mr001)