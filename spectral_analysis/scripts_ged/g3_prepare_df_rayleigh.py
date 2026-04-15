import numpy as np
import pandas as pd
import os
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from spectral_analysis.helper_functions import import_mask_and_parcellation, plot_partial_residuals
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

def prepare_ged_dataframe(df, config, strategy):

    # compute GED for frequencies in specified range
    freq_min = config["ged_min_investigated_frequency"]
    freq_max = config["ged_max_investigated_frequency"]
    freq_step = config["ged_frequency_step_size"]
    frequencies_ged = np.arange(freq_min, freq_max + freq_step, freq_step)
    frequencies_ged = np.round(frequencies_ged, decimals=2)
    frequencies_ged = np.concatenate((frequencies_ged, np.array([0])))
    # frequencies_ged = frequencies_ged[frequencies_ged < 0.2]

    spectrum_top_dir = 'data/ged_results'
    ged_list = []
    for index, scan in df.iterrows():

        spectrum_dir = spectrum_top_dir + '/' + strategy + '/' + scan.subject + '/' + scan.session + '/func/'
        try:
            rayleighs_file = os.path.join(spectrum_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_ged_rayleigh.txt'))
            rayleighs = np.loadtxt(rayleighs_file)
        except:
            print(f"Scan {scan['preproc_filename_cifti']} does not exist, skipping")
            continue
        print(f"Processing scan: {scan.subject} {scan.session} {scan.task} {scan.run}",end='\r')

        # average across voxels 
        for i in range(config['num_eigenvectors_to_keep']):
            for f,frequency in enumerate(frequencies_ged):
                ged_list.append(scan.to_dict() | {
                    'strategy': strategy,
                    'eigenvector': i,
                    'frequency': frequency,
                    'rayleigh': rayleighs[f,i],
                })

    # compute partial residuals for each strategy, frequency, and network
    df_out = pd.DataFrame(ged_list)
    for frequency in df_out['frequency'].unique():
        print(f'Calculating partial residuals for frequency: {frequency}',end='\r')
        for eigenvector in range(config['num_eigenvectors_to_keep']):
            df_stat = df_out[(df_out['strategy'] == strategy) & (df_out['eigenvector'] == eigenvector) & (df_out['frequency'] == frequency)]

            if df_stat.shape[0] != 122:
                raise ValueError(f'Expected 122 scans for frequency {frequency} and eigenvector {eigenvector} but got {df_stat.shape[0]}')

            df_stat = df_stat.rename(columns={'ratio_outliers_fd0.5_std_dvars1000': 'ratio_outliers_fd0_5_std_dvars1000'})

            formula0 = "rayleigh ~ 1 + " + " + ".join(config['nuisance_regressors'])
            m0 = MixedLM.from_formula(formula0, groups="subject", data=df_stat).fit(reml=True)
            partial_residuals = m0.resid + m0.fe_params['Intercept'] # add back the intercept to get the partial residuals on the same scale as the original variable

            formula1 = "rayleigh ~ 1 + " + " + ".join(config['nuisance_regressors_nomotion'])
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
    import os
    with open("config.json") as f:
        config = json.load(f)
    denoising_strategies = config['strategies']
    df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
    df = df[df['task']==config["task"]]
    df = df[df['include_scan_coil_numvols']]
    df = df[df['include_manual_qc']]
    df = df[df['ratio_outliers_fd0.5_std_dvars1000'] < config["max_ratio_outliers_fd0.5_std_dvars1000"]]
    df = df[df['max_fd'] < config["scan_max_fd_threshold"]]
    os.makedirs('data/results', exist_ok=True)

    for strategy in denoising_strategies:
        if strategy not in ['9p']:
            continue
        print(f'Preparing power dataframe for strategy: {strategy}')
        spectrum_df = prepare_ged_dataframe(df, config, strategy)

        spectrum_df.to_csv('data/results/ged_rayleigh_'+strategy+'.csv', index=False)