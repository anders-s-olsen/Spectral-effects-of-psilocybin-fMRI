import nibabel as nib
import numpy as np
import pandas as pd
import os
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
import matplotlib.pyplot as plt

def prepare_connectomes(df, config):
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
    
    subjects = df['subject'].unique()
    rows = []

    parcel_labels = np.loadtxt('data/external/Schaefer2018_200Parcels_17Networks_order_Tian_Subcortex_S2_label.txt', dtype=str, usecols=0)
    parcel_labels = parcel_labels[0::2]
    parcel_labels[:32] = ['Subcort' + label for label in parcel_labels[:32]]
    triu = np.triu_indices(parcel_labels.shape[0], k=1)
    networks = config['networks']
    
    for s, denoising_strategy in enumerate(config['strategies']):
        for b, band in enumerate(config['frequency_bands']):
            for subject in subjects:
                for t, time_interval in enumerate(config['time_intervals']):
                    print(f"Processing strategy: {denoising_strategy}, band: {band}, subject: {subject}, time interval: {time_interval}")
                    df_subject = df[df['subject'] == subject]
                    df_subject = df_subject[df_subject['time_interval'] == time_interval]
                    if df_subject.empty:
                        continue
                    
                    for index, scan in df_subject.iterrows():
                        connectome_dir = 'data/connectomes/' + denoising_strategy + '/' + band + '/' + scan.subject + '/' + scan.session + '/func/'
                        
                        connectome_file = os.path.join(connectome_dir, os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_connectome_'+band+'_parcellated_schaefertian232.txt'))
                        connectome = np.loadtxt(connectome_file)
                        # random_matrix = np.random.rand(parcel_labels.shape[0], parcel_labels.shape[0])
                        # connectome = (random_matrix + random_matrix.T)/2  # make it symmetric

                        # compute within and between network means
                        # avg_connectome = np.zeros((len(networks),len(networks)))
                        for i,roi1 in enumerate(networks):
                            mask1 = np.array([roi1 in label for label in parcel_labels])
                            for j,roi2 in enumerate(networks[i:]):
                                mask2 = np.array([roi2 in label for label in parcel_labels])
                                if roi1==roi2:
                                    connectome_reduced = connectome[np.ix_(mask1,mask2)][np.triu_indices(np.sum(mask1),k=1)]
                                else:
                                    connectome_reduced = connectome[np.ix_(mask1,mask2)]
                                connectome_mean = np.nanmean(connectome_reduced)
                                connectome_mean_fisher = np.nanmean(np.arctanh(connectome_mean))
                                # avg_connectome[i,j] += connectome_mean
                                         
                                row = {
                                    'subject': scan.subject,
                                    'session': scan.session,
                                    'task': scan.task,
                                    'run': scan.run,
                                    'age': scan.age,
                                    'sex': scan.sex,
                                    'MR': scan.scanner,
                                    'PPL': scan['PPL_mcg/L'],
                                    'SDI': scan['SDI'],
                                    'strategy': denoising_strategy,
                                    'time_interval': time_interval,
                                    'frequency': band,
                                    'network_1': roi1,
                                    'network_2': roi2,
                                    'connectivity': connectome_mean,
                                    'connectivity_fisher': connectome_mean_fisher
                                    }
                                rows.append(row)

    df_out = pd.DataFrame(rows)
    for denoising_strategy in config['strategies']:
        for band in config['frequency_bands']:
            for i,network1 in enumerate(networks):
                for network2 in networks[i:]:
                    print(f'Calculating partial residuals for {denoising_strategy} {band}')
                    df_stat = df_out[(df_out['strategy'] == denoising_strategy) & (df_out['frequency'] == band) & (df_out['network_1'] == network1) & (df_out['network_2'] == network2)]
                    m0 = MixedLM.from_formula("connectivity ~ 1 + " + " + ".join(config['nuisance_regressors']), groups="subject", data=df_stat).fit(reml=True)
                    # Design matrix under null
                    X0 = m0.model.exog[:,1:]
                    beta0 = m0.params.values[1:-1]
                    partial_residuals = df_stat['connectome_mean_fisher'] - X0 @ np.atleast_1d(beta0)
                    # check that theres nothing in the residuals row/col yet
                    if 'partial_residuals_age-sex-scanner' in df_out.columns:
                        if df_out.loc[df_stat.index,'partial_residuals_age-sex-scanner'].notnull().any():
                            raise ValueError('Residuals column already contains data')
                    df_out.loc[df_stat.index,'partial_residuals_age-sex-scanner'] = partial_residuals
    return df_out

if __name__ == "__main__":
    
    import json
    with open("config.json") as f:
        config = json.load(f)
    df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
    df = df[df['task']==config["task"]]
    df = df[df['include_scan_coil_numvols']]
    df = df[df['include_manual_qc']]
    df = df[df['ratio_outliers_fd0.5_std_dvars1.5'] < config["ratio_outliers_fd0.5_std_dvars1.5"]]

    spectrum_df_agg = prepare_connectomes(df, config)
    spectrum_df_agg.to_csv('data/results/connectomes_by_band_networks.csv')