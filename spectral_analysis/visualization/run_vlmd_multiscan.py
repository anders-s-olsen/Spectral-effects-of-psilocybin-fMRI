# %%
import numpy as np
import pandas as pd
# from scripts_spectral_compute.mvmd import mvmd
from vlmd import vlmd   
from vlmd_multiscan import vlmd_multiscan
import matplotlib.pyplot as plt
import json
import os
with open("config.json") as f:
    config = json.load(f)

# %%
strategies = config["strategies"]
strategy = strategies[-1]

# Load the DataFrame containing scan information
df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
df = df[df['task']==config["task"]]
df = df[df['include_scan_coil_numvols']]
df = df[df['include_manual_qc']]
df = df[df['ratio_outliers_fd0.5_std_dvars1.5'] < config["ratio_outliers_fd0.5_std_dvars1.5"]]

data = []
fs = []
stds = []
scanners = []
for idx, scan in df.iterrows():
    print(f'Loading data for subject {scan.subject}, session {scan.session}, task {scan.task}')
    image_path = 'data/denoised/'+strategy+'/' + scan.subject + '/' + scan.session + '/func/'
    image_file = os.path.basename(scan.preproc_filename_cifti.replace('.dtseries.nii', '_denoised_parcellated_schaefertian232.txt'))
    data_scan = np.loadtxt(image_path+'/'+image_file)
    data.append(data_scan.T)
    fs.append(1 / scan.tr)
    stds.append(np.std(data_scan))
    scanners.append(scan.scanner)

fs = np.array(fs)
stds = np.array(stds)
scanners = np.array(scanners)

# Z-score per scanner
for scanner in np.unique(scanners):
    idxs = np.where(scanners == scanner)[0]
    data_scanner = np.concatenate([data[i] for i in idxs], axis=1)
    std = np.std(data_scanner)
    for i in idxs:
        data[i] /= std

# %%
num_modes = 6
num_latents = 5
alpha = 1
reg_lambda = 0.001
reg_rho = 1
tolerance = 1e-8
max_iter = 2000
modes, latents, omega, modes_hat = vlmd_multiscan(data, 
                                        num_modes=num_modes, 
                                        num_latents=num_latents,
                                        alpha=alpha, 
                                        reg_lambda=reg_lambda,
                                        reg_rho=reg_rho,
                                        tolerance=tolerance, 
                                        sampling_rate=fs, 
                                        max_iter=max_iter, 
                                        verbose=True)

# numpy save all the results in one variable
np.savez('data/results/vlmd_multiscan_'+strategy+'_modes-'+str(num_modes)+'_latents-'+str(num_latents)+'_lambda='+str(reg_lambda)+'.npz', 
         latents=latents, omega=omega, modes_hat=modes_hat, 
         alpha=alpha, reg_lambda=reg_lambda, reg_rho=reg_rho, tolerance=tolerance, max_iter=max_iter)

a = 8
