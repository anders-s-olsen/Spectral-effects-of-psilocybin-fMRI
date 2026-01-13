import numpy as np
import os
import pandas as pd

def compute_connectomes(df,config):
    for index, scan in df.iterrows():
        for denoising_strategy in config['strategies']:
            denoised_dir = 'data/denoised/'+denoising_strategy+'/' + scan.subject + '/' + scan.session + '/func/'
            for band in config['frequency_bands']:
                print(f"Processing scan: {scan.subject} {scan.run} with strategy {denoising_strategy} and band {band}")
                data = np.loadtxt(denoised_dir + os.path.basename(scan['preproc_filename_cifti']).replace(
                    '.dtseries.nii', '_denoised_filtered_'+band+'_parcellated_schaefertian232.txt'))
                connectome = np.corrcoef(data.T)
                out_dir = 'data/connectomes/' + denoising_strategy + '/' + band + '/' + scan.subject + '/' + scan.session + '/func/'
                os.makedirs(out_dir, exist_ok=True)
                out_file = out_dir + os.path.basename(scan['preproc_filename_cifti']).replace('.dtseries.nii', '_connectome_'+band+'_parcellated_schaefertian232.txt')
                np.savetxt(out_file, connectome)

if __name__ == "__main__":
    import json

    with open("config.json") as f:
        config = json.load(f)

    # Load the DataFrame containing scan information
    df = pd.read_csv('data/func_scans_table_outliers_ses-PSI_PPLSDI.csv')
    df = df[df['task']==config["task"]]
    
    # Compute spectra
    compute_connectomes(df, config)