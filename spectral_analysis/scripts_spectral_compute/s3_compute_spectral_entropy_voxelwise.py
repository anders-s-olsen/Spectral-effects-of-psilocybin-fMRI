import pandas as pd
import os 
import nibabel as nib
import numpy as np

def spectral_entropy(df, denoising_strategy, downsample_mr001=False):
    """
    Compute Shannon entropy.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing scan information.
    despiking (bool): If True, apply despiking to the time series.
    
    Returns:
    None
    """

    # find denoising directories
    spectrum_top_dir = 'data/spectra/'+denoising_strategy

    for index, scan in df.iterrows():
        print(f"Processing scan: {scan.subject} {scan.session} {scan.task} {scan.run}")

        if downsample_mr001 and scan.scanner == 'MR001':
            add_downsampled_label = '_downsampled'
        else:
            add_downsampled_label = ''

        spectrum_dir = spectrum_top_dir + '/' + scan.subject + '/' + scan.session + '/func/'

        try:
            spectrum_file = os.path.join(spectrum_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', add_downsampled_label+'_mtspectra.dtseries.nii'))
            # check if the output file already exists because this can take a long time
            if os.path.exists(spectrum_file.replace('.dtseries.nii', '_entropy.dscalar.nii')):
                print(f"Output file already exists for scan {os.path.basename(scan['preproc_filename_cifti'])}, skipping")
                continue
            img = nib.load(spectrum_file)
        except:
            print(f"Scan {scan['preproc_filename_cifti']} does not exist, skipping")
            continue

        # Load the scan
        data = img.get_fdata()

        if scan.scanner == 'MR001' and not downsample_mr001:
            frequencies = np.loadtxt('data/frequencies_MR001.txt')
        elif scan.scanner == 'MR45' or downsample_mr001:
            frequencies = np.loadtxt('data/frequencies_MR45.txt')

        # dead voxels set to nan
        data[:,(data**2).sum(axis=0) == 0] = np.nan
        # compute spectral entropy (Shannon entropy) between 0.01-0.2Hz
        freq_mask = (frequencies >= 0.01) & (frequencies <= 0.2)
        data_masked = data[freq_mask, :]
        data_masked_norm = data_masked / np.nansum(data_masked, axis=0)  # normalize to sum to 1

        entropy = -np.sum(data_masked_norm * np.log2(data_masked_norm), axis=0) #bits

        # Save the time series to a file
        output_file = spectrum_file.replace('.dtseries.nii', '_entropy.dscalar.nii')
        scalar_axis = nib.cifti2.ScalarAxis(['Spectral entropy']) 
        time_axis, brain_model_axis = [img.header.get_axis(i) for i in range(img.ndim)]
        new_header = nib.Cifti2Header.from_axes([scalar_axis, brain_model_axis])
        nib.save(nib.Cifti2Image(entropy[np.newaxis,:], new_header), output_file)

if __name__ == "__main__":
    import json

    with open("config.json") as f:
        config = json.load(f)

    strategies = config["strategies"]

    # Load the DataFrame containing scan information
    df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
    df = df[df['task']==config["task"]]

    downsample_mr001 = False

    for denoising_strategy in strategies:
        print(f"Computing spectral entropy for denoising strategy: {denoising_strategy}")
        spectral_entropy(df, denoising_strategy, downsample_mr001=downsample_mr001)