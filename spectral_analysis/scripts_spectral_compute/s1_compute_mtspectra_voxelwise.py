import numpy as np
import os
import nibabel as nib
import nitime.algorithms as tsa
import pandas as pd

def compute_spectra(df,denoising_strategy, downsample_mr001=False):

    for index, scan in df.iterrows():

        if downsample_mr001 and scan.scanner == 'MR001':
            # Parameters for downsampling
            TR_fast = 0.8  # Original TR for MR001
            TR_slow = 2.0  # Desired TR to match MR45
            add_downsampled_label = '_downsampled'
        else:
            add_downsampled_label = ''

        denoised_dir = 'data/denoised/'+denoising_strategy+'/' + scan.subject + '/' + scan.session + '/func/'
        try:
            # check if the output file already exists because this can take a long time
            output_dir = 'data/spectra/'+denoising_strategy+'/' + scan.subject + '/' + scan.session + '/func/'
            if os.path.exists(output_dir+ os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', add_downsampled_label+'_mtspectra.dtseries.nii')):
                print(f"Output file already exists for scan {os.path.basename(scan['preproc_filename_cifti'])}, skipping")
                continue
            print(f"Processing scan: {scan.subject} {scan.session} {scan.task} {scan.run}")
            img_file = os.path.join(denoised_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_denoised.dtseries.nii'))
            img = nib.load(img_file)
        except:
            print(f"Scan {scan['preproc_filename_cifti']} does not exist, skipping")
            continue

        # Load the scan
        data = img.get_fdata()
        orig_num_voxels = data.shape[1]
        active_voxels = np.sum(data**2, axis=0) > 0
        non_artifactual_voxels = np.any(data>config['voxel_max_psc_threshold'],axis=0) == False
        from scipy.stats import kurtosis
        low_kurtosis_voxels = kurtosis(data,axis=0) <= config['voxel_max_kurtosis_threshold']
        mask = active_voxels & non_artifactual_voxels & low_kurtosis_voxels
        data = data[:, mask]  # only keep active voxels
        # data[:,(data**2).sum(axis=0) == 0] = np.nan # dead voxels set to nan
        
        if downsample_mr001 and scan.scanner == 'MR001':
            n_fast = data.shape[0]
            t_fast = np.arange(n_fast) * TR_fast    # times of fast samples: 0, 0.8, 1.6, ...
            t_slow = np.arange(0, t_fast[-1] + TR_slow, TR_slow)  # 0, 2, 4, ...
            # # pick nearest index for each slow time
            # idx = np.argmin(np.abs(t_fast[:,None] - t_slow[None,:]), axis=0)
            # data_decimated = data[idx]
            # # Linear interpolation function
            from scipy.interpolate import interp1d
            interp_func = interp1d(t_fast, data, axis=0, kind='linear',
                                bounds_error=False, fill_value="extrapolate")
            data_linear = interp_func(t_slow)
            data = data_linear
        
        if scan.scanner == 'MR001' and not downsample_mr001:
            NFFT = 1000
            TR = 2.0
        elif scan.scanner == 'MR45' or downsample_mr001:
            NFFT = 400
            TR = 0.8

        f, psd_mt, _ = tsa.multi_taper_psd(data.T, #assumes data is space x time
                                            Fs=1/TR,
                                            NW=None, # defaults to 4, which means 8 tapers
                                            BW=None, # defaults to None
                                            adaptive=False, # adaptive weighting of tapers, could be used (slow)
                                            jackknife=False, # jackknife estimation of variance, which we don't assess
                                            low_bias=True, # Only use tapers with low bias, which is the default
                                            sides='default', #always onesided for non-complex-valued data
                                            NFFT=NFFT)  
        psd_mt_all = np.empty((len(f), orig_num_voxels))
        psd_mt_all[:] = np.nan
        psd_mt_all[:, mask] = psd_mt.T  # only fill in active voxels
        delta_f = f[1] - f[0]  # frequency resolution

        # save the PSD as cifti file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', add_downsampled_label+'_mtspectra.dtseries.nii'))
        
        series_axis = nib.cifti2.SeriesAxis(start=0,step=delta_f,size=len(f),unit='hertz') 
        time_axis, brain_model_axis = [img.header.get_axis(i) for i in range(img.ndim)]
        new_header = nib.Cifti2Header.from_axes([series_axis, brain_model_axis])
        nib.save(nib.Cifti2Image(psd_mt_all, new_header), output_file)

        # also save f as a text file
        if scan.scanner == 'MR001' and not downsample_mr001:
            if not os.path.exists('data/frequencies_MR001.txt'):
                np.savetxt('data/frequencies_MR001.txt', f)
        elif scan.scanner == 'MR45':
            if not os.path.exists('data/frequencies_MR45.txt'):
                np.savetxt('data/frequencies_MR45.txt', f)

if __name__ == "__main__":
    import json

    with open("config.json") as f:
        config = json.load(f)

    strategies = config["strategies"]

    # Load the DataFrame containing scan information
    df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
    df = df[df['task']==config["task"]]
    df = df[df['include_manual_qc']]
    df = df[df['include_scan_coil_numvols']]

    downsample_mr001 = False
    
    # Compute spectra
    for denoising_strategy in strategies:
        print(f"Computing spectra for denoising strategy: {denoising_strategy}")
        compute_spectra(df, denoising_strategy, downsample_mr001=downsample_mr001)