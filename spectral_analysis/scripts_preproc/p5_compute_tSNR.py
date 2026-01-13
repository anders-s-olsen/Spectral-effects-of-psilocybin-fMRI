import pandas as pd
import os 
import nibabel as nib
import numpy as np
from spectral_analysis.helper_functions import import_mask_and_parcellation

def compute_tsnr(df):
    # load parcellation
    _, parcellation, _, _ = import_mask_and_parcellation()
    unique_parcels = np.unique(parcellation)
    unique_parcels = unique_parcels[unique_parcels != 0]  # remove background parcel
    # find denoising directories
    preproc_top_dir = 'data/preprocessed/'

    for index, scan in df.iterrows():
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

        # dead voxels set to nan
        data[:,(data**2).sum(axis=0) == 0] = np.nan

        tsnr = np.nanmean(data, axis=0) / np.nanstd(data, axis=0)

        # Save the time series to a file
        output_file = preproc_file.replace('.dtseries.nii', '_tsnr.dscalar.nii')
        scalar_axis = nib.cifti2.ScalarAxis(['tSNR']) 
        time_axis, brain_model_axis = [img.header.get_axis(i) for i in range(img.ndim)]
        new_header = nib.Cifti2Header.from_axes([scalar_axis, brain_model_axis])
        nib.save(nib.Cifti2Image(tsnr[np.newaxis,:], new_header), output_file)        
        
if __name__ == "__main__":

    import json
    with open("config.json") as f:
        config = json.load(f)

    strategies = config["strategies"]

    df = pd.read_csv('data/func_scans_table_outliers.csv')
    df = df[df['session']==config["session"]]
    df = df[df['task']==config["task"]]       
    
    compute_tsnr(df)