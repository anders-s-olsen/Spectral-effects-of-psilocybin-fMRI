import numpy as np
import pandas as pd
import os
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from spectral_analysis.helper_functions import import_mask_and_parcellation, plot_partial_residuals
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

def prepare_power_dataframe(df, config, strategy, downsample_mr001=False):

    # parcels to networks mapping
    parcel_labels, _, _, _ = import_mask_and_parcellation()

    bins_per_group = 8  # Number of bins to group together in the aggregated spectrum for statistical testing

    spectrum_top_dir = 'data/spectra'
    spectrum_list = []
    spectrum_list_agg = []

    for index, scan in df.iterrows():

        if downsample_mr001 and scan.scanner == 'MR001':
            add_downsampled_label = '_downsampled'
            # both correspond to mr45 frequencies after downsampling
            frequencies = {'MR45': np.loadtxt('data/frequencies_MR45.txt'), 'MR001': np.loadtxt('data/frequencies_MR45.txt')}            
        else:
            add_downsampled_label = ''
            frequencies = {'MR45': np.loadtxt('data/frequencies_MR45.txt'), 'MR001': np.loadtxt('data/frequencies_MR001.txt')}

        spectrum_dir = spectrum_top_dir + '/' + strategy + '/' + scan.subject + '/' + scan.session + '/func/'
        try:
            spectrum_file = os.path.join(spectrum_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', add_downsampled_label+'_mtspectra_parcellated_schaefertian232.txt'))
            parcellated_spectrum = np.loadtxt(spectrum_file)
        except:
            print(f"Scan {scan['preproc_filename_cifti']} does not exist, skipping")
            continue

        # average across voxels 
        for i,network in enumerate(config['networks']):
            mask = np.array([network in label for label in parcel_labels])
            power = np.nanmean(parcellated_spectrum[:, mask], axis=1)
            for f,frequency in enumerate(frequencies[scan.scanner]):
                spectrum_list.append(scan.to_dict() | {
                    'strategy': strategy,
                    'network': network,
                    'frequency': frequency,
                    'log_power': power[f],
                    #'log_power': np.log10(power[f])
                })
            
            power_agg = power[1:(power.shape[0]//bins_per_group)*bins_per_group+1].reshape(-1, bins_per_group).mean(axis=1)
            frequencies_agg = frequencies[scan.scanner][1:(power.shape[0]//bins_per_group)*bins_per_group+1].reshape(-1, bins_per_group).mean(axis=1)
            for f,frequency in enumerate(frequencies_agg):
                spectrum_list_agg.append(scan.to_dict() | {
                    'strategy': strategy,
                    'network': network,
                    'frequency': frequency,
                    'log_power': power_agg[f],
                    #'log_power': np.log10(power_agg[f])
                })

    # compute partial residuals for each strategy, frequency, and network
    df_out = pd.DataFrame(spectrum_list)
    for frequency in df_out['frequency'].unique():
        if frequency == 0 or frequency > config["max_investigated_frequency"]:
            continue
        for network in config['networks']:
            print(f'Calculating partial residuals for {strategy} {frequency} {network}')
            df_stat = df_out[(df_out['strategy'] == strategy) & (df_out['network'] == network) & (df_out['frequency'] == frequency)]

            df_stat = df_stat.rename(columns={'ratio_outliers_fd0.5_std_dvars1000': 'ratio_outliers_fd0_5_std_dvars1000'})

            m0 = MixedLM.from_formula(config['target_variable'] + " ~ 1 + " + " + ".join(config['nuisance_regressors']), groups="subject", data=df_stat).fit(reml=True)
            X0 = m0.model.exog[:,1:] # exclude intercept
            beta0 = m0.fe_params.values[1:] 
            partial_residuals = df_stat[config['target_variable']] - X0 @ np.atleast_1d(beta0)
            df_stat = df_stat.assign(partial_residuals=partial_residuals)

            m0 = MixedLM.from_formula(config['target_variable'] + " ~ 1 + " + " + ".join(config['nuisance_regressors_nomotion']), groups="subject", data=df_stat).fit(reml=True)
            X0 = m0.model.exog[:,1:] # exclude intercept
            beta0 = m0.fe_params.values[1:] 
            partial_residuals_nomotion = df_stat[config['target_variable']] - X0 @ np.atleast_1d(beta0)
            
            # m01 = MixedLM.from_formula(config['target_variable'] + " ~ 1 + " + " + scanner", groups="subject", data=df_stat).fit(reml=True)
            # X01 = m01.model.exog[:,1:] # exclude intercept
            # beta01 = m01.fe_params.values[1:] 
            # partial_residuals_onlyscanner = df_stat[config['target_variable']] - X01 @ np.atleast_1d(beta01)
            # df_stat = df_stat.assign(partial_residuals_onlyscanner=partial_residuals_onlyscanner)

            # if np.any(np.isclose(frequency,config['frequencies_to_plot_partial_residuals'])):
            #     plot_partial_residuals(df_stat, target_variable=config['target_variable'], savename='network_frequency/'+config['target_variable']+'_partial_residuals_'+strategy+'_'+network+'_freq'+str(frequency))
            
            # check that theres nothing in the residuals row/col yet
            if 'partial_residuals' in df_out.columns:
                if df_out.loc[df_stat.index,'partial_residuals'].notnull().any():
                    raise ValueError('Residuals column already contains data')
            df_out.loc[df_stat.index,'partial_residuals'] = partial_residuals
            df_out.loc[df_stat.index,'partial_residuals_nomotion'] = partial_residuals_nomotion
    return df_out, pd.DataFrame(spectrum_list_agg)

if __name__ == "__main__":
    
    import json
    import os
    with open("config.json") as f:
        config = json.load(f)
    df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
    df = df[df['task']==config["task"]]
    df = df[df['include_scan_coil_numvols']]
    df = df[df['include_manual_qc']]
    denoising_strategies = config['strategies']
    df = df[df['ratio_outliers_fd0.5_std_dvars1000'] < config["max_ratio_outliers_fd0.5_std_dvars1000"]]
    df = df[df['max_fd'] < config["scan_max_fd_threshold"]]

    downsample_mr001 = False
    if downsample_mr001:
        add_downsampled_label = '_downsampled'
    else:
        add_downsampled_label = ''

    for strategy in denoising_strategies:
        if not strategy in ['high-pass-only','high-pass-motion','acompcor','9p']:
            continue
        print(f'Preparing power dataframe for strategy: {strategy}')
        spectrum_df, spectrum_df_agg = prepare_power_dataframe(df, config, strategy, downsample_mr001=downsample_mr001)

        spectrum_df.to_csv('data/results/spectra_by_frequency_network_'+strategy+add_downsampled_label+'.csv', index=False)
        # only some columns should be exported
        spectrum_df_agg = spectrum_df_agg[spectrum_df_agg['frequency'] < config["max_investigated_frequency"]]
        spectrum_df_agg.to_csv('data/results/spectra_by_frequency_network_agg_'+strategy+add_downsampled_label+'.csv', index=False)