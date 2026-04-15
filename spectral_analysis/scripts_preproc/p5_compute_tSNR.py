import pandas as pd
import os 
import nibabel as nib
import numpy as np
from scipy.stats import kurtosis

def compute_tsnr(df):
    # find denoising directories
    preproc_top_dir = 'data/preprocessed/'

    for index, scan in df.iterrows():
        # if int(scan.subject[4:]) <= 56165:
        #     continue
        print(f"Processing scan: {scan.subject} {scan.session} {scan.task} {scan.run}")

        preproc_dir = preproc_top_dir + '/' + scan.subject + '/' + scan.session + '/func/'
        try:
            preproc_file = os.path.join(preproc_dir, os.path.basename(scan['preproc_filename_cifti']))
        except:
            print(f"Scan {scan['preproc_filename_cifti']} does not exist, skipping")
            continue

        # Load the scan
        img = nib.load(preproc_file)
        data = img.get_fdata()

        # convert to percent signal change
        mean_signal = np.nanmean(data, axis=0)
        data_pct = (data - mean_signal) / mean_signal * 100

        data[:,np.any(data_pct>config['voxel_max_psc_threshold'],axis=0)] = np.nan  # extreme values set to nan
        data[:,kurtosis(data_pct,axis=0)>config['voxel_max_kurtosis_threshold']] = np.nan  # extreme kurtosis set to nan

        # dead voxels set to nan
        data[:,(data**2).sum(axis=0) == 0] = np.nan
        data[:,np.nanstd(data, axis=0) == 0] = np.nan

        tsnr = np.nanmean(data, axis=0) / np.nanstd(data, axis=0)

        # Save the time series to a file
        output_file = preproc_file.replace('.dtseries.nii', '_tsnr.dscalar.nii')
        scalar_axis = nib.cifti2.ScalarAxis(['tSNR']) 
        time_axis, brain_model_axis = [img.header.get_axis(i) for i in range(img.ndim)]
        new_header = nib.Cifti2Header.from_axes([scalar_axis, brain_model_axis])
        nib.save(nib.Cifti2Image(tsnr[np.newaxis,:], new_header), output_file)        

        # make a bids-valid json
        json_file = output_file.replace('.dscalar.nii', '.json')
        json_dict = {
            "Description": "Temporal signal-to-noise ratio (tSNR) map computed as mean signal across time divided by standard deviation across time for each voxel/vertex. Dead voxels with zero variance across time are set to NaN.",
            "Units": "arbitrary",
            "LongName": "Temporal Signal-to-Noise Ratio",
            "Source": preproc_file
        }
        with open(json_file, 'w') as f:
            json.dump(json_dict, f, indent=4)
        
if __name__ == "__main__":

    import json
    with open("config.json") as f:
        config = json.load(f)

    df = pd.read_csv('data/func_scans_table_outliers.csv')
    # df = df[df['session']==config["session"]]
    # df = df[df['task']==config["task"]]       
    
    compute_tsnr(df)