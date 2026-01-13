from glob import glob
import nilearn.interfaces.fmriprep.load_confounds_utils
import pandas as pd
import nilearn.interfaces
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def extract_confounds_append_outliers(df, denoising_strategy, fd_threshold=None, std_dvars_threshold=None, ratio_outlier_exclusion=0.15, spike_regression=False):
    """
    Extract confounds and append outliers to the DataFrame.
    """
    subjects = df['subject'].unique()
    for subject in subjects:
        subject_df = df[df['subject'] == subject]
        print(f"Processing subject: {subject}")
        
        if 'aroma' in denoising_strategy:
            # find preprocessed scans
            preprocessed_scans = subject_df['preproc_filename_cifti_aroma'].dropna().tolist()
        else:
            preprocessed_scans = subject_df['preproc_filename_cifti'].dropna().tolist()
        
        if denoising_strategy == 'save-outliers-in-table-only':
            if fd_threshold is None or std_dvars_threshold is None or ratio_outlier_exclusion is None:
                raise ValueError("fd_threshold, std_dvars_threshold, and ratio_outlier_exclusion must be provided for 'save-outliers-in-table-only' strategy")
            confounds = nilearn.interfaces.fmriprep.load_confounds(
                preprocessed_scans, 
                strategy=['motion','scrub'], 
                motion='basic', #basic=6p
                fd_threshold=fd_threshold, 
                std_dvars_threshold=std_dvars_threshold,
                scrub=0) # no scrubbing, just get outliers
        elif 'acompcor' in denoising_strategy:
            confounds = nilearn.interfaces.fmriprep.load_confounds(
                preprocessed_scans,
                strategy=('motion','compcor','high_pass'), #highpass must be included for aCompCor
                motion='derivatives', #full=motion24, derivaties=muschelli2014 (12p)
                compcor='anat_separated', # [Muschelli2014]
                n_compcor=5,
                demean=False)
        elif '9p' in denoising_strategy:
            confounds = nilearn.interfaces.fmriprep.load_confounds(
                preprocessed_scans,
                strategy=('motion','wm_csf','global_signal','high_pass'), 
                motion='basic', #basic=6p
                wm_csf='basic', # basic=mean WM, mean CSF
                global_signal='basic', # basic=mean global signal
                demean=False)  
        elif denoising_strategy == 'ica-aroma':
            confounds = nilearn.interfaces.fmriprep.load_confounds(
                preprocessed_scans,
                strategy=('ica_aroma','wm_csf','high_pass'), 
                ica_aroma='full', # use fMRIPrep output ~desc-smoothAROMAnonaggr_bold.nii.gz.
                wm_csf='basic', # basic=mean WM, mean CSF
                demean=False)  
        elif denoising_strategy == 'ica-aroma-gsr':
            confounds = nilearn.interfaces.fmriprep.load_confounds(
                preprocessed_scans,
                strategy=('ica_aroma','wm_csf','global_signal','high_pass'), 
                ica_aroma='full', # use fMRIPrep output ~desc-smoothAROMAnonaggr_bold.nii.gz.
                wm_csf='basic', # basic=mean WM, mean CSF
                global_signal='basic', # basic=mean global signal
                demean=False) 
        elif denoising_strategy == 'high-pass-only':
            confounds = nilearn.interfaces.fmriprep.load_confounds(
                preprocessed_scans,
                strategy=('high_pass',), 
                demean=False)
        elif denoising_strategy == 'high-pass-motion':
            confounds = nilearn.interfaces.fmriprep.load_confounds(
                preprocessed_scans,
                strategy=('high_pass','motion'), 
                motion='derivatives', #full=motion24, derivaties=muschelli2014 (12p)
                demean=False)
        else:
            raise ValueError(f"Unknown strategy: {denoising_strategy}") 

        for i,scan in enumerate(preprocessed_scans):
            confounds_file_orig = nilearn.interfaces.fmriprep.load_confounds_utils.get_confounds_file(scan,flag_full_aroma=False) 
            if denoising_strategy == 'save-outliers-in-table-only':
                confounds_file = pd.read_csv(confounds_file_orig, sep='\t')
                fd = confounds_file['framewise_displacement'].values
                mean_fd = np.nanmean(fd)
                max_fd = np.nanmax(fd)
                if max_fd > config["scan_max_fd_threshold"]:
                    print(f"Consider excluding scan {os.path.basename(scan)} due to high max FD: {max_fd:.2f}")
                std_dvars = confounds_file['std_dvars'].values
                mean_std_dvars = np.nanmean(std_dvars)

                # scan_length = confounds[0][i].shape[0]
                # all_samples = np.arange(scan_length)
                # motion_outlier = np.isin(all_samples, confounds[1][i], invert=True)
                motion_outlier = (fd > fd_threshold) | (std_dvars > std_dvars_threshold)
                scan_length = len(motion_outlier)
                
                # calculate the number of outliers
                num_outliers = motion_outlier.sum()
                ratio = num_outliers / scan_length
                if ratio > ratio_outlier_exclusion:
                    print(f"Consider excluding scan {os.path.basename(scan)} due to high ratio of outliers: {ratio:.2f}")
                # add the ratio of outliers to the DataFrame
                df.loc[df['preproc_filename_cifti'] == scan, 'ratio_outliers_fd'+str(fd_threshold)+'_std_dvars'+str(std_dvars_threshold)] = ratio
                df.loc[df['preproc_filename_cifti'] == scan, 'mean_fd'] = mean_fd
                df.loc[df['preproc_filename_cifti'] == scan, 'mean_std_dvars'] = mean_std_dvars
                df.loc[df['preproc_filename_cifti'] == scan, 'max_fd'] = max_fd
                df.loc[df['preproc_filename_cifti'] == scan, 'outlier_locs'] = ','.join(map(str, np.where(motion_outlier)[0].tolist()))
                
                if spike_regression:
                    # include these columns in the DataFrame as columns called "motion_outlier_00", "motion_outlier_01", etc.
                    # one hot encode the motion outlier
                    motion_outlier_one_hot = np.zeros((scan_length, motion_outlier.sum()), dtype=int)
                    motion_outlier_one_hot[motion_outlier, np.arange(motion_outlier.sum())] = 1
                    for j in range(motion_outlier_one_hot.shape[1]):
                        confounds[0][i][f'motion_outlier{j:02d}'] = motion_outlier_one_hot[:, j]
                
            else:
                # Save the updated confounds DataFrame to a new CSV file
                
                if spike_regression:   
                    output_file = confounds_file_orig.replace('preprocessed','interim/confounds_'+denoising_strategy+'_spike_regression_fd'+str(fd_threshold)+'_std_dvars'+str(std_dvars_threshold)).replace('.tsv', '_filtered.tsv')
                else:
                    output_file = confounds_file_orig.replace('preprocessed','interim/confounds_'+denoising_strategy).replace('.tsv', '_filtered.tsv')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                confounds[0][i].to_csv(output_file, index=False, sep='\t')
    return df

if __name__ == "__main__":

    import json
    with open("config.json") as f:
        config = json.load(f)

    # denoising strategies:
    strategies = config["strategies"] + ['save-outliers-in-table-only']
    spike_regression = config["spike_regression"]

    # Here we can change fd_threshold and std_dvars_threshold and num_outlier_exclusion criteria
    # These will be included in the dataframe, not the confounds file
    volume_fd_threshold = config["volume_fd_threshold"]
    volume_std_dvars_threshold = config["volume_std_dvars_threshold"]

    # We can load the scans table from the previous step to add a new set of columns for outlier detection with altered thresholds
    # As a failsafe, all .csv files are deleted in steps 1) and 2) to avoid having copies of the same table with different information in them.
    if os.path.exists('data/func_scans_table_outliers.csv'):
        df = pd.read_csv('data/func_scans_table_outliers.csv')
    else:
        df = pd.read_csv('data/func_scans_table.csv')

    for denoising_strategy in strategies:
        # if denoising_strategy != 'save-outliers-in-table-only':
        #     continue
        print(f"Extracting confounds and appending outliers for strategy: {denoising_strategy}")

        if spike_regression:
            #remove existing files to avoid appending to them
            existing_files = glob('data/interim/confounds_'+denoising_strategy+'_spike_regression_fd'+str(volume_fd_threshold)+'_std_dvars'+str(volume_std_dvars_threshold)+'/sub-*/ses-*/func/*'+denoising_strategy+'.tsv', recursive=True)
            for file in existing_files:
                os.remove(file)
        else:
            #remove existing files to avoid appending to them
            existing_files = glob('data/interim/confounds_'+denoising_strategy+'/sub-*/ses-*/func/*.tsv')
            for file in existing_files:
                os.remove(file)

        # Extract confounds and append outliers to the DataFrame
        df = extract_confounds_append_outliers(df, denoising_strategy, volume_fd_threshold, volume_std_dvars_threshold, ratio_outlier_exclusion=0.15, spike_regression=False)

        # Save the updated DataFrame to a new CSV file
        if denoising_strategy == 'save-outliers-in-table-only':
            output_file = 'data/func_scans_table_outliers.csv'
            df.to_csv(output_file, index=False)