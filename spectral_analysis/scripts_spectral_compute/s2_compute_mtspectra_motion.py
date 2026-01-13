import numpy as np
import os
import pandas as pd
import nilearn.interfaces
import nitime.algorithms as tsa
import nilearn.signal

def compute_spectra(df, downsample_mr001=False):

    for index, scan in df.iterrows():

        if downsample_mr001 and scan.scanner == 'MR001':
            # Parameters for downsampling
            TR_fast = 0.8  # Original TR for MR001
            TR_slow = 2.0  # Desired TR to match MR45
            add_downsampled_label = '_downsampled'
        else:
            add_downsampled_label = ''

        output_dir = 'data/spectra/motion/' + scan.subject + '/' + scan.session + '/func/'
        
        try:
            confounds_file_orig = nilearn.interfaces.fmriprep.load_confounds_utils.get_confounds_file(scan['preproc_filename_cifti'], flag_full_aroma=False)
        except:
            print(f"Scan {scan['preproc_filename_cifti']} does not exist, skipping")
            continue
        confounds = pd.read_csv(confounds_file_orig, sep='\t')
        confounds_dct = confounds.filter(like='cosine')

        if confounds_dct.empty:
            print(f"No DCT confounds found for scan {scan['preproc_filename_cifti']}, skipping")
            continue

        motion = confounds[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].values

        # Detrend the confounds
        motion_denoised = nilearn.signal.clean(motion,
                                confounds=confounds_dct,
                                t_r=scan.tr,
                                filter=False,
                                detrend=True,
                                standardize=False,
                                standardize_confounds=True,
                                ensure_finite=True) 
        
        if downsample_mr001 and scan.scanner == 'MR001':
            n_fast = motion_denoised.shape[0]
            t_fast = np.arange(n_fast) * TR_fast    # times of fast samples: 0, 0.8, 1.6, ...
            t_slow = np.arange(0, t_fast[-1] + TR_slow, TR_slow)  # 0, 2, 4, ...
            # # pick nearest index for each slow time
            # idx = np.argmin(np.abs(t_fast[:,None] - t_slow[None,:]), axis=0)
            # data_decimated = data[idx]
            # # Linear interpolation function
            from scipy.interpolate import interp1d
            interp_func = interp1d(t_fast, motion, axis=0, kind='linear',
                                bounds_error=False, fill_value="extrapolate")
            motion_denoised_linear = interp_func(t_slow)
            motion_denoised = motion_denoised_linear

        if scan.scanner == 'MR001' and not downsample_mr001:
            NFFT = 1000
            TR = 0.8
        elif scan.scanner == 'MR45' or downsample_mr001:
            NFFT = 400
            TR = 2

        f, psd_mt, _ = tsa.spectral.multi_taper_psd(motion_denoised.T, #assumes data is space x time
                                                    Fs=1/TR,
                                                    NW=None, # defaults to 4, which means 8 tapers
                                                    BW=None, # defaults to None
                                                    adaptive=False, # adaptive weighting of tapers, could be used (slow)
                                                    jackknife=False, # jackknife estimation of variance, which we don't assess
                                                    low_bias=True, # Only use tapers with low bias, which is the default
                                                    sides='default', #always onesided for non-complex-valued data
                                                    NFFT=NFFT)  
        # Save the spectrum as txt
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', add_downsampled_label+'_motion_mtspectra.txt')), psd_mt.T)
        np.savetxt(os.path.join(output_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', add_downsampled_label+'_motion.txt')), motion_denoised.T)
        
if __name__ == "__main__":
    import json

    with open("config.json") as f:
        config = json.load(f)

    strategies = config["strategies"]

    # Load the DataFrame containing scan information
    df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
    df = df[df['task']==config["task"]]

    downsample_mr001 = False
    
    compute_spectra(df, downsample_mr001=downsample_mr001)