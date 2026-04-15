import numpy as np
import pandas as pd
import os
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from spectral_analysis.helper_functions import import_mask_and_parcellation, plot_partial_residuals
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

def prepare_power_maps(df, config, strategy):
    """
    Prepare power maps for different strategies and time intervals.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing scan information.
    
    Returns:
    spectrum_df (pd.DataFrame): DataFrame containing power maps for each scan.
    spectrum_df_agg (pd.DataFrame): Aggregated DataFrame for power maps.
    """
    
    spectrum_top_dir = 'data/spectra'
    rows = []

    parcel_labels, _, _, _ = import_mask_and_parcellation(config['parcellation'])
    frequencies = {'MR45': np.loadtxt('data/frequencies_MR45.txt'), 'MR001': np.loadtxt('data/frequencies_MR001.txt')}

    for index, scan in df.iterrows():
        spectrum_dir = spectrum_top_dir + '/' + strategy + '/' + scan.subject + '/' + scan.session + '/func/'
        try:
            spectrum_file = os.path.join(spectrum_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_mtspectra_parcellated_'+config['parcellation']+'.txt'))
            spectrum = np.loadtxt(spectrum_file)
        except:
            # print(f"Scan {scan['preproc_filename_cifti']} does not exist, skipping")
            continue
        print(f"Processing scan: {scan.subject} {scan.session} {scan.task} {scan.run}",end='\r')
        
        for i,band in enumerate(config['frequency_bands']):
            high_pass, low_pass = config['frequency_bands'][band]
            frequencies_idx = (frequencies[scan.scanner] >= high_pass) & (frequencies[scan.scanner] <= low_pass)
            power = np.nanmean(spectrum[frequencies_idx,:], axis=0)

            for roi in range(power.shape[0]):
                if np.isnan(power[roi]):
                    power[roi] = np.mean(power[~np.isnan(power)])  # replace nan with mean power
                    print(f'Warning: Replaced nan power value for subject {scan.subject}, session {scan.session}, band {band}, roi {parcel_labels[roi]} with mean power')
                row = scan.to_dict() | {
                    'strategy': strategy,
                    'band': band,
                    'roi':parcel_labels[roi],
                    'log_power': power[roi],
                    #'log_power': np.log10(power[roi])
                }
                rows.append(row)

    df_out = pd.DataFrame(rows)
    for band in config['frequency_bands']:
        for i,roi_label in enumerate(parcel_labels):
            if 'Subcort' in roi_label:
                continue
            print(f'Calculating partial residuals for {strategy} {band} {roi_label}',end='\r')
            df_stat = df_out[(df_out['roi'] == roi_label) & (df_out['band'] == band)]
            df_stat = df_stat.rename(columns={'ratio_outliers_fd0.5_std_dvars1000': 'ratio_outliers_fd0_5_std_dvars1000'})

            if df_stat.shape[0] != 125:
                raise ValueError(f'Expected 125 scans for band {band} and roi {roi_label} but got {df_stat.shape[0]}')

            formula0 = config['target_variable'] + " ~ 1 + " + " + ".join(config['nuisance_regressors'])
            m0 = MixedLM.from_formula(formula0, groups="subject", data=df_stat).fit(reml=True)
            partial_residuals = m0.resid + m0.fe_params['Intercept'] # add back the intercept to get the partial residuals on the same scale as the original variable

            formula1 = config['target_variable'] + " ~ 1 + " + " + ".join(config['nuisance_regressors_nomotion'])
            m1 = MixedLM.from_formula(formula1, groups="subject", data=df_stat).fit(reml=True)
            partial_residuals_nomotion = m1.resid + m1.fe_params['Intercept']

            # check that theres nothing in the residuals row/col yet
            if 'partial_residuals' in df_out.columns:
                if df_out.loc[df_stat.index,'partial_residuals'].notnull().any():
                    raise ValueError('Residuals column already contains data')
            df_out.loc[df_stat.index,'partial_residuals'] = partial_residuals
            df_out.loc[df_stat.index,'partial_residuals_nomotion'] = partial_residuals_nomotion
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
    os.makedirs('data/results', exist_ok=True)

    for strategy in denoising_strategies:
        if strategy not in ['9p']:
            continue        
        spectrum_df_agg = prepare_power_maps(df, config, strategy)
        spectrum_df_agg.to_csv('data/results/spectra_by_band_parcel_'+strategy+'.csv')