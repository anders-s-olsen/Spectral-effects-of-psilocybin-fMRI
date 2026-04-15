import numpy as np
import pandas as pd
import os
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
from spectral_analysis.helper_functions import import_mask_and_parcellation, plot_partial_residuals
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
import matplotlib.pyplot as plt
import seaborn as sns
def prepare_power_dataframe(df, config, strategy, downsample_mr001=False):

    # compute GED for frequencies in specified range
    freq_min = config["min_statistics_frequency"]
    freq_max = config["max_statistics_frequency"]
    freq_step = config["statistics_frequency_step_size"]
    frequencies_stats = np.arange(freq_min, freq_max + freq_step, freq_step)
    frequencies_stats = np.round(frequencies_stats, decimals=2)

    # parcels to networks mapping
    parcel_labels, _, _, _ = import_mask_and_parcellation(config['parcellation'])

    # bins_per_group = 8  # Number of bins to group together in the aggregated spectrum for statistical testing

    spectrum_top_dir = 'data/spectra'
    spectrum_list = []
    spectrum_list_agg = []

    if downsample_mr001:
        frequencies = {'MR45': np.loadtxt('data/frequencies_MR45.txt'), 'MR001': np.loadtxt('data/frequencies_MR45.txt')}   
    else:
        frequencies = {'MR45': np.loadtxt('data/frequencies_MR45.txt'), 'MR001': np.loadtxt('data/frequencies_MR001.txt')}
    freq_filter = {}
    for scanner in ['MR45','MR001']:
        frequencies[scanner] = np.round(frequencies[scanner], decimals=5)
        freq_filter[scanner] = np.zeros((len(frequencies_stats),len(frequencies[scanner])))
        for f,frequency in enumerate(frequencies_stats):
            freq_filter[scanner][f] = np.exp(-0.5 * ((frequencies[scanner] - frequency)/(freq_step))**2)
            freq_filter[scanner][f] /= np.sum(freq_filter[scanner][f])  # normalize filter to sum to 1

    for index, scan in df.iterrows():

        if downsample_mr001 and scan.scanner == 'MR001':
            add_downsampled_label = '_downsampled'     
        else:
            add_downsampled_label = ''

        spectrum_dir = spectrum_top_dir + '/' + strategy + '/' + scan.subject + '/' + scan.session + '/func/'
        try:
            spectrum_file = os.path.join(spectrum_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', add_downsampled_label+'_mtspectra_parcellated_'+config['parcellation']+'.txt'))
            parcellated_spectrum = np.loadtxt(spectrum_file)
        except:
            print(f"Scan {scan['preproc_filename_cifti']} does not exist, skipping")
            continue
        print(f"Processing scan: {scan.subject} {scan.session} {scan.task} {scan.run}",end='\r')

        # average across voxels 
        for i,network in enumerate(config['networks']):
            mask = np.array([network in label for label in parcel_labels])
            power = np.nanmean(parcellated_spectrum[:, mask], axis=1)
            for f,frequency in enumerate(frequencies[scan.scanner]):
                spectrum_list.append(scan.to_dict() | {
                    'strategy': strategy,
                    'network': network,
                    'frequency': frequency,
                    'log_power': power[f], # 10log10 has already been taken during parcellation
                    #'log_power': np.log10(power[f])
                })
            
            for f,frequency in enumerate(frequencies_stats):
                power_filtered = np.sum(power * freq_filter[scan.scanner][f])
                spectrum_list_agg.append(scan.to_dict() | {
                    'strategy': strategy,
                    'network': network,
                    'frequency': frequency,
                    'log_power': power_filtered,
                    #'log_power': np.log10(power_agg[f])
                })

    # compute partial residuals for each strategy, frequency, and network
    df_out = pd.DataFrame(spectrum_list)
    for frequency in df_out['frequency'].unique():
        print(f'Calculating partial residuals for frequency: {frequency}',end='\r')
        if frequency == 0 or frequency > config["nyquist"]:
            continue
        # if frequency not in frequencies_stats:
        #     continue
        for network in config['networks']:
            df_stat = df_out[(df_out['strategy'] == strategy) & (df_out['network'] == network) & (df_out['frequency'] == frequency)]

            if df_stat.shape[0] != 125:
                raise ValueError(f'Expected 125 scans for frequency {frequency} and network {network} but got {df_stat.shape[0]}')

            df_stat = df_stat.rename(columns={'ratio_outliers_fd0.5_std_dvars1000': 'ratio_outliers_fd0_5_std_dvars1000'})

            formula0 = config['target_variable'] + " ~ 1 + " + " + ".join(config['nuisance_regressors'])
            m0 = MixedLM.from_formula(formula0, groups="subject", data=df_stat).fit(reml=True)
            partial_residuals = m0.resid + m0.fe_params['Intercept'] # add back the intercept to get the partial residuals on the same scale as the original variable

            formula1 = config['target_variable'] + " ~ 1 + " + " + ".join(config['nuisance_regressors_nomotion'])
            m1 = MixedLM.from_formula(formula1, groups="subject", data=df_stat).fit(reml=True)
            partial_residuals_nomotion = m1.resid + m1.fe_params['Intercept']

            # df_stat['PPL_mcg_L'] = df_stat['PPL_mcg/L']
            # df_stat.dropna(inplace=True, subset=['PPL_mcg_L'])
            # formula2 = config['target_variable'] + " ~ 1 + PPL_mcg_L +" + " + ".join(config['nuisance_regressors'])
            # m2 = MixedLM.from_formula(formula2, groups="subject", data=df_stat).fit(reml=True)
            # residuals = m2.resid
            # df_stat['residuals'] = residuals
            # df_stat['fittedvalues'] = m2.fittedvalues

            # # make some plots on residuals. residuals vs fitted colored by time and with lines for each subject, residuals vs time_interval as spaghetti plot colored by subject, and histogram of residuals
            # plt.figure(figsize=(15,5))
            # plt.subplot(1,3,1)
            # sns.scatterplot(x='fittedvalues', y='residuals', hue='time_interval', data=df_stat, color='viridis')
            # plt.legend(title='Time interval',loc='upper center')
            # plt.title(f'Residuals vs fitted values')
            # plt.xlabel('Fitted values')
            # plt.ylabel('Residuals')
            # plt.subplot(1,3,2)
            # sns.lineplot(x='time_interval', y=residuals, hue='subject', data=df_stat, legend=False)
            # plt.title(f'Residuals vs time interval (color is subject)')
            # plt.xlabel('Time interval')
            # plt.ylabel('Residuals')
            # plt.subplot(1,3,3)
            # plt.hist(residuals, bins=20)
            # plt.title(f'Histogram of residuals')
            # plt.xlabel('Residuals')
            # plt.ylabel('Count')
            # plt.tight_layout()
            # os.makedirs('figures/diagnostics', exist_ok=True)
            # plt.savefig(f'figures/diagnostics/residuals_powervsfrequency_{strategy}_network-{network}_frequency-{frequency:.2f}.png')
            # plt.close()
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
    denoising_strategies = config['strategies']
    df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
    df = df[df['task']==config["task"]]
    df = df[df['include_scan_coil_numvols']]
    df = df[df['include_manual_qc']]
    df = df[df['ratio_outliers_fd0.5_std_dvars1000'] < config["max_ratio_outliers_fd0.5_std_dvars1000"]]
    df = df[df['max_fd'] < config["scan_max_fd_threshold"]]
    os.makedirs('data/results', exist_ok=True)

    downsample_mr001 = False
    if downsample_mr001:
        add_downsampled_label = '_downsampled'
    else:
        add_downsampled_label = ''

    for strategy in denoising_strategies:
        if strategy not in ['9p']:
            continue
        print(f'Preparing power dataframe for strategy: {strategy}')
        spectrum_df, spectrum_df_agg = prepare_power_dataframe(df, config, strategy, downsample_mr001=downsample_mr001)

        spectrum_df.to_csv('data/results/spectra_by_frequency_network_'+strategy+add_downsampled_label+'.csv', index=False)
        # only some columns should be exported
        spectrum_df_agg.to_csv('data/results/spectra_by_frequency_network_agg_'+strategy+add_downsampled_label+'.csv', index=False)