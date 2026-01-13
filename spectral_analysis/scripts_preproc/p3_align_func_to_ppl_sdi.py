import pandas as pd
import numpy as np
import warnings
from glob import glob
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

import json
with open("config.json") as f:
    config = json.load(f)

scan_time_intervals = config["time_intervals"]

# delete existing files to avoid appending to them
existing_files = glob('data/*PPLSDI*.csv')
for file in existing_files:
    os.remove(file)

# read the PPL and SDI data
ppl_sdi = pd.read_excel('data/SDI_PPL_P3_20220202.xlsx')
age_sex = pd.read_excel('data/CIMBI_age_sex_clean.xlsx')

# convert timestamps to datetime, fill in missing values for SDI with those for PPL
ppl_sdi['Clock_time'] = pd.to_datetime(ppl_sdi['Clock_time'], errors='coerce', format='%H:%M:%S')
ppl_sdi['SDI_clocktime_if_more_than_10_mins_diff_to_clock_time'] = pd.to_datetime(ppl_sdi['SDI_clocktime_if_more_than_10_mins_diff_to_clock_time'], errors='coerce', format='%H:%M:%S')
ppl_sdi['SDI_clocktime_if_more_than_10_mins_diff_to_clock_time'].fillna(ppl_sdi['Clock_time'], inplace=True)

# we need to convert times into datetimes to work with them:
ppl_time_df = pd.DataFrame()
ppl_time_df['year'] = ppl_sdi['MR_scan_date'].dt.year
ppl_time_df['month'] = ppl_sdi['MR_scan_date'].dt.month
ppl_time_df['day'] = ppl_sdi['MR_scan_date'].dt.day
ppl_time_df['hour'] = ppl_sdi['Clock_time'].dt.hour
ppl_time_df['minute'] = ppl_sdi['Clock_time'].dt.minute
ppl_time_df['second'] = ppl_sdi['Clock_time'].dt.second
sdi_time_df = pd.DataFrame()
sdi_time_df['year'] = ppl_sdi['MR_scan_date'].dt.year
sdi_time_df['month'] = ppl_sdi['MR_scan_date'].dt.month
sdi_time_df['day'] = ppl_sdi['MR_scan_date'].dt.day
sdi_time_df['hour'] = ppl_sdi['SDI_clocktime_if_more_than_10_mins_diff_to_clock_time'].dt.hour
sdi_time_df['minute'] = ppl_sdi['SDI_clocktime_if_more_than_10_mins_diff_to_clock_time'].dt.minute
sdi_time_df['second'] = ppl_sdi['SDI_clocktime_if_more_than_10_mins_diff_to_clock_time'].dt.second

ppl_sdi['ppl_time'] = pd.to_datetime(ppl_time_df)
ppl_sdi['sdi_time'] = pd.to_datetime(sdi_time_df)

# read the functional scans table
func_scans = pd.read_csv('data/func_scans_table_outliers.csv')
func_scans = func_scans[func_scans['session']=='ses-PSI']

subjects = func_scans['subject'].unique()
for subject in subjects:
    ppl_sdi_subject = ppl_sdi[ppl_sdi['CIMBI.ID'] == int(subject[4:])]
    meas_date = pd.to_datetime(ppl_sdi_subject['MR_scan_date'])
    drug_date = pd.to_datetime(meas_date.values[0])
    
    # check that only one drug administration time is present
    drug_admin_time = ppl_sdi_subject['drug_admin_time'].values
    if len(np.unique(drug_admin_time)) != 1:
        raise ValueError(f"Multiple drug administration times found for subject {subject}: {drug_admin_time}")

    # combine date and time into a single timestamp
    drug_admin_timestamp = drug_date.replace(hour=drug_admin_time[0].hour, minute=drug_admin_time[0].minute, second=drug_admin_time[0].second)
    if subject == 'sub-56145':
        # the 'ppl_time' before 8 in the morning should have their date advanced by one day
        ppl_next_day_id = ppl_sdi_subject['ppl_time'].dt.hour < 8
        sdi_next_day_id = ppl_sdi_subject['sdi_time'].dt.hour < 8
        ppl_sdi_subject.loc[ppl_next_day_id, 'ppl_time'] += pd.Timedelta(days=1)
        ppl_sdi_subject.loc[sdi_next_day_id, 'sdi_time'] += pd.Timedelta(days=1)
 
    ppl_sdi_subject['ppl_time_since_admin'] = ((ppl_sdi_subject['ppl_time'] - drug_admin_timestamp).dt.total_seconds() / 60).round()
    ppl_sdi_subject['sdi_time_since_admin'] = ((ppl_sdi_subject['sdi_time'] - drug_admin_timestamp).dt.total_seconds() / 60).round()
    
    # check for times since admin above 30 mins and ppl = 0
    time30_ppl0 = (ppl_sdi_subject['ppl_time_since_admin'] > 30) & (ppl_sdi_subject['psi_conc_mcg_per_L'] == 0)
    if time30_ppl0.sum() > 0:
        print(f"Warning: Subject {subject} has PPL measurements with time since admin > 30 minutes and value 0. These will be set to NaN since they are likely a mistake.")
        ppl_sdi_subject.loc[time30_ppl0, 'psi_conc_mcg_per_L'] = np.nan

    subject_scans = func_scans[func_scans['subject'] == subject]
    # for scan in range(len(subject_scans)):
    for idx, scan in subject_scans.iterrows():

        func_scans.at[idx, 'age'] = age_sex.loc[age_sex['Cimbi'] == int(subject[4:]), 'Age'].values[0]
        func_scans.at[idx, 'sex'] = age_sex.loc[age_sex['Cimbi'] == int(subject[4:]), 'Sex'].values[0]

        scan_start_timestamp = pd.to_datetime(scan['scan_start_time'])
        scan_end_timestamp = pd.to_datetime(scan['scan_end_time'])
        scan_mid_timestamp = scan_start_timestamp + (scan_end_timestamp - scan_start_timestamp) / 2
        # throw away the microsecond part for brevity
        scan_mid_timestamp = scan_mid_timestamp.replace(microsecond=0)

        # count minutes after drug administration
        scan_time_since_drug_admin = np.round((scan_mid_timestamp - drug_admin_timestamp).total_seconds()/60)
        func_scans.at[idx, 'scan_min_since_admin'] = scan_time_since_drug_admin
        func_scans.at[idx, 'drug_admin_time'] = drug_admin_timestamp

        if scan_time_since_drug_admin < 0 and scan.subject=='sub-55772':
            # these subjects have no PPL or SDI measurements before the drug administration
            func_scans.at[idx, 'PPL_mcg/L'] = 0
            func_scans.at[idx, 'ppl_min_since_scan'] = 0
            func_scans.at[idx, 'ppl_min_since_admin'] = 0
            func_scans.at[idx, 'ppl_time'] = pd.NaT
            func_scans.at[idx, 'SDI'] = 0
            func_scans.at[idx, 'sdi_min_since_scan'] = 0
            func_scans.at[idx, 'sdi_min_since_admin'] = 0
            func_scans.at[idx, 'sdi_time'] = pd.NaT
            continue

        ppl_time_since_scan = ((ppl_sdi_subject['ppl_time'] - scan_mid_timestamp).dt.total_seconds()/60).round()
        sdi_time_since_scan = ((ppl_sdi_subject['sdi_time'] - scan_mid_timestamp).dt.total_seconds()/60).round()
        # find the lowest positive value
        if scan_time_since_drug_admin < 0:
            # if the scan was before the drug administration, we can only use the measurements before the drug administration
            ppl_time_since_scan = ppl_time_since_scan[ppl_sdi_subject['ppl_time_since_admin'] < 0]
            sdi_time_since_scan = sdi_time_since_scan[ppl_sdi_subject['sdi_time_since_admin'] < 0]
        else:
            # find the lowest positive value in the ppl_meas_time_since_scan where there is a measurement in the psi_conc_mcg_per_L column
            ppl_time_since_scan = ppl_time_since_scan[ppl_sdi_subject['psi_conc_mcg_per_L'].notna()]
            sdi_time_since_scan = sdi_time_since_scan[ppl_sdi_subject['SDI_score'].notna()]
        
        closest_ppl_time_since_scan = ppl_time_since_scan.abs().min()
        closest_ppl_argmin = ppl_time_since_scan.abs().idxmin()
        closest_sdi_time_since_scan = sdi_time_since_scan.abs().min()
        closest_sdi_argmin = sdi_time_since_scan.abs().idxmin()

        # if the scan was before the drug administration but no SDI measurement was taken, set SDI to 0
        if scan_time_since_drug_admin < 0 and np.isnan(ppl_sdi_subject['SDI_score'][closest_sdi_argmin]):
            SDI = 0
        else:
            SDI = ppl_sdi_subject['SDI_score'][closest_sdi_argmin]

        skip_ppl = False
        skip_sdi = False
        if scan.task == 'task-rest':
            if np.abs(ppl_time_since_scan[closest_ppl_argmin])>100:
                print(f"Warning: Scan {scan['run']} for subject {subject} has a PPL measurement more than 100 minutes away from the scan time. Closest PPL measurement since scan: {ppl_time_since_scan[closest_ppl_argmin]} minutes.")
                print(f"Excluding ppl for scan {scan['run']} for subject {subject}.")
                func_scans.at[idx, 'PPL_mcg/L'] = np.nan
                func_scans.at[idx, 'ppl_min_since_scan'] = np.nan
                func_scans.at[idx, 'ppl_min_since_admin'] = np.nan
                func_scans.at[idx, 'ppl_time'] = pd.NaT
                skip_ppl = True
            elif np.abs(sdi_time_since_scan[closest_sdi_argmin])>100:
                print(f"Warning: Scan {scan['run']} for subject {subject} has an SDI measurement more than 100 minutes away from the scan time. Closest SDI measurement since scan: {sdi_time_since_scan[closest_sdi_argmin]} minutes.")
                print(f"Excluding sdi for scan {scan['run']} for subject {subject}.")
                func_scans.at[idx, 'SDI'] = np.nan
                func_scans.at[idx, 'sdi_min_since_scan'] = np.nan
                func_scans.at[idx, 'sdi_min_since_admin'] = np.nan
                func_scans.at[idx, 'sdi_time'] = pd.NaT
                skip_sdi = True

            elif scan_time_since_drug_admin > 0 and scan_time_since_drug_admin < 200:
                if np.abs(ppl_time_since_scan[closest_ppl_argmin]) > 20:
                    print(f"Warning: Scan {scan['run']} for subject {subject} has a PPL measurement more than 20 minutes away from the scan time. Closest PPL measurement since scan: {ppl_time_since_scan[closest_ppl_argmin]} minutes.")
                    if subject in ['sub-55992','sub-57142'] and np.abs(ppl_time_since_scan[closest_ppl_argmin]) < 30:
                        print(f"Accepting scan {scan['run']} for subject {subject} despite the warning, since the ppl trajectory matches the other measurements.")
                    else:
                        print(f"Excluding ppl for scan {scan['run']} for subject {subject}.")
                        func_scans.at[idx, 'PPL_mcg/L'] = np.nan
                        func_scans.at[idx, 'ppl_min_since_scan'] = np.nan
                        func_scans.at[idx, 'ppl_min_since_admin'] = np.nan
                        func_scans.at[idx, 'ppl_time'] = pd.NaT
                        skip_ppl = True

                if np.abs(sdi_time_since_scan[closest_sdi_argmin]) > 20:
                    print(f"Warning: Scan {scan['run']} for subject {subject} has a SDI measurement more than 20 minutes away from the scan time. Closest SDI measurement since scan: {sdi_time_since_scan[closest_sdi_argmin]} minutes.")
                    if subject in ['sub-55992','sub-56145','sub-57142','sub-57193'] and np.abs(sdi_time_since_scan[closest_sdi_argmin]) < 30:
                        # these subjects are accepted anyways
                        print(f"Accepting scan {scan['run']} for subject {subject} despite the warning, since the sdi trajectory matches the other measurements.")
                    else:
                        print(f"Excluding sdi for scan {scan['run']} for subject {subject}.")
                        func_scans.at[idx, 'SDI'] = np.nan
                        func_scans.at[idx, 'sdi_min_since_scan'] = np.nan
                        func_scans.at[idx, 'sdi_min_since_admin'] = np.nan
                        func_scans.at[idx, 'sdi_time'] = pd.NaT
                        skip_sdi = True

        # add the corresponding measurement to the scan
        if not skip_ppl:
            func_scans.at[idx, 'PPL_mcg/L'] = ppl_sdi_subject['psi_conc_mcg_per_L'][closest_ppl_argmin]
            func_scans.at[idx, 'ppl_min_since_scan'] = ppl_time_since_scan[closest_ppl_argmin]
            func_scans.at[idx, 'ppl_min_since_admin'] = ppl_sdi_subject['ppl_time_since_admin'][closest_ppl_argmin]
            func_scans.at[idx, 'ppl_time'] = ppl_sdi_subject['ppl_time'][closest_ppl_argmin]
        if not skip_sdi:
            func_scans.at[idx, 'SDI'] = SDI
            func_scans.at[idx, 'sdi_min_since_scan'] = sdi_time_since_scan[closest_sdi_argmin]
            func_scans.at[idx, 'sdi_min_since_admin'] = ppl_sdi_subject['sdi_time_since_admin'][closest_sdi_argmin]
            func_scans.at[idx, 'sdi_time'] = ppl_sdi_subject['sdi_time'][closest_sdi_argmin]

        # add the correct time interval for the scan
        for interval, time_range in scan_time_intervals.items():
            if time_range[0] <= scan_time_since_drug_admin < time_range[1]:
                func_scans.at[idx, 'time_interval'] = interval
                break

# save the updated scans for the subject
# reorder the columns for easier reading
func_scans = func_scans[['subject', 'session', 'task', 'run','age','sex',
                         'drug_admin_time','ppl_time','sdi_time',
                         'scan_start_time','scan_end_time', 'scan_min_since_admin', 'ppl_min_since_admin','sdi_min_since_admin',
                        'ppl_min_since_scan','sdi_min_since_scan','PPL_mcg/L', 'SDI', 'time_interval',
                        'tr', 'te','num_vols','ped', 'coil_name', 'coil_active', 'scanner', 'include_scan_coil_numvols','include_manual_qc',
                        'ratio_outliers_fd'+str(0.5)+'_std_dvars'+str(1000),'mean_fd', 'mean_std_dvars','max_fd','outlier_locs',
                        'scan_filename','preproc_filename_volumetric', 'preproc_filename_cifti','preproc_filename_cifti_despiked','preproc_filename_cifti_aroma']]
func_scans.to_csv(f'data/func_scans_table_outliers_ses-PSI_PPLSDI.csv', index=False)
func_scans_rest = func_scans[func_scans['task'] == 'task-rest']
func_scans_rest.to_csv(f'data/func_scans_table_outliers_ses-PSI_task-rest_PPLSDI.csv', index=False)
print("All subjects processed.")

        
        
    
