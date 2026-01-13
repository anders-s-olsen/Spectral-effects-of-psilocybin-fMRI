import nibabel as nib
import numpy as np
import pandas as pd
import os
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

def prepare_power_maps(df, config, strategy):
    """
    Prepare power maps for different strategies and time intervals.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing scan information.
    strategies (list): List of denoising strategies.
    time_intervals (list): List of time intervals to process.
    
    Returns:
    spectrum_df (pd.DataFrame): DataFrame containing power maps for each scan.
    spectrum_df_agg (pd.DataFrame): Aggregated DataFrame for power maps.
    """
    
    spectrum_top_dir = 'data/spectra'
    subjects = df['subject'].unique()
    rows = []
    frequencies = {'MR45': np.loadtxt('data/frequencies_MR45.txt'), 'MR001': np.loadtxt('data/frequencies_MR001.txt')}

    for index, scan in df.iterrows():
        print(f"Processing strategy: {strategy}, subject: {scan.subject}, session: {scan.session}")
        spectrum_dir = spectrum_top_dir + '/' + strategy + '/' + scan.subject + '/' + scan.session + '/func/'
        try:
            spectrum_file = os.path.join(spectrum_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_mtspectra.dtseries.nii'))
            spectrum = nib.load(spectrum_file).get_fdata()
        except:
            # print(f"Scan {scan['preproc_filename_cifti']} does not exist, skipping")
            continue

        for i,band in enumerate(config['frequency_bands']):
            high_pass, low_pass = config['frequency_bands'][band]
            
            power = np.nanmean(spectrum[(frequencies[scan.scanner] >= high_pass) & (frequencies[scan.scanner] <= low_pass),:], axis=0)
            for voxel in range(power.shape[0]):
                row = scan.to_dict() | {
                    'strategy': strategy,
                    'band': band,
                    'voxel': voxel,
                    'power': power[voxel],
                    'log_power': 10*np.log10(power[voxel])
                }
                rows.append(row)

    # compute partial residuals for each strategy, frequency, and network
    df_out = pd.DataFrame(rows)
    for band in config['frequency_bands']:
        for voxel in range(power.shape[0]):
            print(f'Calculating partial residuals for {strategy} {band} {voxel}')

            df_stat = df_out[(df_out['voxel'] == voxel) & (df_out['band'] == band)]
            m0 = MixedLM.from_formula(config['target_variable'] + " ~ 1 + " + " + ".join(config['nuisance_regressors']), groups="subject", data=df_stat).fit(reml=True)
            # Design matrix under null
            X0 = m0.model.exog[:,1:] # exclude intercept
            beta0 = m0.fe_params.values[1:] 
            partial_residuals = df_stat[config['target_variable']] - X0 @ np.atleast_1d(beta0)
            
            # check that theres nothing in the residuals row/col yet
            if 'partial_residuals' in df_out.columns:
                if df_out.loc[df_stat.index,'partial_residuals'].notnull().any():
                    raise ValueError('Residuals column already contains data')
            df_out.loc[df_stat.index,'partial_residuals'] = partial_residuals

    return df_out

if __name__ == "__main__":
    
    import json
    with open("config.json") as f:
        config = json.load(f)
    df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
    df = df[df['task']==config["task"]]
    df = df[df['include_scan_coil_numvols']]
    df = df[df['include_manual_qc']]
    df = df[df['ratio_outliers_fd0.5_std_dvars1000'] < config["max_ratio_outliers_fd0.5_std_dvars1000"]]
    df = df[df['max_fd'] < config["scan_max_fd_threshold"]]
    denoising_strategies = config['strategies']

    for strategy in denoising_strategies:
        print(f'Preparing power dataframe for strategy: {strategy}')

        spectrum_df_agg = prepare_power_maps(df, config, strategy)
        spectrum_df_agg.to_csv('data/results/spectra_by_band_voxels_'+strategy+'.csv')