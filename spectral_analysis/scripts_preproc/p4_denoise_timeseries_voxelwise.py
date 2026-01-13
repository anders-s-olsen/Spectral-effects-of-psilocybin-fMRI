import nilearn.signal
import nilearn.interfaces
import pandas as pd
import os 
import nibabel as nib
import numpy as np

def denoise_data(df, denoising_strategy, bands=None, spike_regression=False, fd_threshold=None, std_dvars_threshold=None, standardize=True):
        
    for index, scan in df.iterrows():

        # Load the scan
        input_dir = 'data/preprocessed/' + scan.subject + '/' + scan.session + '/func/'
        try:
            # check if the output file already exists because this can take a long time
            output_dir = 'data/denoised/'+denoising_strategy+'/' + scan.subject + '/' + scan.session + '/func/'
            if os.path.exists(output_dir+ os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_denoised.dtseries.nii')):
                print(f"Output file already exists for scan {scan['preproc_filename_cifti']}, skipping")
                continue
            print(f"Processing scan: {scan.subject} {scan.session} {scan.task} {scan.run}")
            if 'despike' in denoising_strategy:
                img = nib.load(os.path.join(input_dir, os.path.basename(scan['preproc_filename_cifti']).replace('_bold.dtseries.nii', '_desc-despiked_bold.dtseries.nii')))
            elif 'aroma' in denoising_strategy:
                img = nib.load(os.path.join(input_dir, os.path.basename(scan['preproc_filename_cifti']).replace('_bold.dtseries.nii', '_desc-smoothAROMAnonaggr_bold.dtseries.nii')))
            else:
                img = nib.load(scan['preproc_filename_cifti'])

            # Load the confounds
            confounds_file_orig = nilearn.interfaces.fmriprep.load_confounds_utils.get_confounds_file(scan['preproc_filename_cifti'], flag_full_aroma=False)
            if spike_regression:
                confounds_file = 'data/interim/confounds_'+denoising_strategy+'_spike_regression_fd'+str(fd_threshold)+'_std_dvars'+str(std_dvars_threshold)+'/'+scan.subject+'/'+scan.session+'/func/' + os.path.basename(confounds_file_orig).replace('.tsv', '_filtered.tsv')
            else:
                confounds_file = 'data/interim/confounds_'+denoising_strategy+'/'+scan.subject+'/'+scan.session+'/func/' + os.path.basename(confounds_file_orig).replace('.tsv', '_filtered.tsv')
            confounds = pd.read_csv(confounds_file, sep='\t')
        except:
            print(f"Scan {scan['preproc_filename_cifti']} does not exist, skipping")
            continue

        data = img.get_fdata()            
        orig_num_voxels = data.shape[1]
        active_voxels = np.sum(data**2, axis=0) > 0
        data = data[:, active_voxels]  # only keep active voxels to speed up processing

        # data = data - np.mean(data, axis=0)  # Demean the data
        if np.sum(np.isnan(data)) > 0:
            print(f"Warning: Scan {scan['preproc_filename_cifti']} contains NaNs, skipping")
        
        # Detrend the confounds
        denoised_data = nilearn.signal.clean(data,
                                confounds=confounds,
                                t_r=scan.tr,
                                filter=False,
                                detrend=True, #DCT does the detrending automatically but nilearn warns when not detrending, doesn't hurt to do it again
                                standardize=standardize,
                                standardize_confounds=False, # they are already standardized
                                ensure_finite=True) # if true, nans will be replaced with zeros - we already add a check for zeros later
        
        os.makedirs(output_dir, exist_ok=True)

        denoised_data_all = np.zeros((denoised_data.shape[0], orig_num_voxels))
        denoised_data_all[:, active_voxels] = denoised_data  # only fill in active voxels

        if spike_regression:
            output_file = os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_denoised_spike_regression_fd'+str(fd_threshold)+'_std_dvars'+str(std_dvars_threshold) + '.dtseries.nii'))
        else:
            output_file = os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_denoised.dtseries.nii'))
        
        nib.save(nib.Cifti2Image(denoised_data_all, img.header), output_file)
        
        # Apply bandpass filtering if specified
        # The bandpass fitering is done after denoising, since the denoising step includes DCT regressors. 
        # According to Lindquist (2019), filtering should also be applied to regressors, but we cant do that, since that would mean low-pass filtering the DCT regressors.
        # According to Lindquist, the most important thing is to apply HIGHpass filtering jointly with confound regression, which we have already done. 
        if bands is not None:
            for band in bands:
                high_pass, low_pass = bands[band]
                filtered_data = nilearn.signal.butterworth(denoised_data,
                                                        sampling_rate=1/scan.tr,
                                                        low_pass=low_pass,
                                                        high_pass=high_pass,
                                                        order=5,
                                                        padtype="odd",
                                                        padlen=None,
                                                        copy=True)
                filtered_data_all = np.zeros((filtered_data.shape[0], orig_num_voxels))
                filtered_data_all[:, active_voxels] = filtered_data  # only fill in active
                if spike_regression:
                    output_file = os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_denoised_spike_regression_fd'+str(fd_threshold)+'_std_dvars'+str(std_dvars_threshold)+ '_filtered_'+band+'.dtseries.nii'))
                else:
                    output_file = os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_denoised_filtered_'+band+'.dtseries.nii'))
                nib.save(nib.Cifti2Image(filtered_data_all, img.header), output_file)

if __name__ == "__main__":

    import json
    with open("config.json") as f:
        config = json.load(f)

    bands = config["frequency_bands"]
    spike_regression = config["spike_regression"]
    strategies = config["strategies"]
    standardize = config["standardize"]

    if spike_regression:
        fd_threshold = config["fd_threshold"]
        std_dvars_threshold = config["std_dvars_threshold"]
    else:
        fd_threshold = None
        std_dvars_threshold = None

    df = pd.read_csv('data/func_scans_table_outliers.csv')
    df = df[df['session']==config["session"]]
    df = df[df['task']==config["task"]]

    # Denoise the data
    for denoising_strategy in strategies:
        print(f"Denoising data with strategy: {denoising_strategy}")
        denoise_data(df, denoising_strategy, bands, spike_regression=spike_regression, fd_threshold=fd_threshold, std_dvars_threshold=std_dvars_threshold,standardize=standardize)