import pandas as pd
import numpy as np
import warnings
from glob import glob
import os
import json
with open('config.json', 'r') as f:
    config = json.load(f)
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

alignment_mode = 'closest_after_scan' # options: 'closest_from_midscan', 'closest_after_scan'

# delete existing files to avoid appending to them
existing_files = glob('data/*PPLSDI*.csv')
for file in existing_files:
    os.remove(file)

# read the PPL and SDI data
age_sex = pd.read_excel('data/CIMBI_age_sex_clean.xlsx')
ppl_sdi = pd.read_excel('data/SDI_PPL_P3_20220202.xlsx')
ppl_sdi['psi_conc_mcg_per_L'] = 1000*1.04*ppl_sdi['psi_conc'] #some values were specified wrongly....
ket = pd.read_excel('data/NP2P3_Ketanserin_MKM_17_03.xlsx')

# convert timestamps to datetime, fill in missing values for SDI with those for PPL
ppl_sdi['Clock_time'] = pd.to_datetime(ppl_sdi['Clock_time'], errors='coerce', format='%H:%M:%S')
ppl_sdi['SDI_clocktime_if_more_than_10_mins_diff_to_clock_time'] = pd.to_datetime(ppl_sdi['SDI_clocktime_if_more_than_10_mins_diff_to_clock_time'], errors='coerce', format='%H:%M:%S')
ppl_sdi['SDI_clocktime_if_more_than_10_mins_diff_to_clock_time'] = ppl_sdi['SDI_clocktime_if_more_than_10_mins_diff_to_clock_time'].fillna(ppl_sdi['Clock_time'])

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

ket['Time'] = pd.to_datetime(ket['Time'], errors='coerce', format='%H:%M:%S')
ket_time_df = pd.DataFrame()
ket_time_df['year'] = ket['First day of intervention'].dt.year
ket_time_df['month'] = ket['First day of intervention'].dt.month
ket_time_df['day'] = ket['First day of intervention'].dt.day
ket_time_df['hour'] = ket['Time'].dt.hour
ket_time_df['minute'] = ket['Time'].dt.minute
ket_time_df['second'] = ket['Time'].dt.second
ket['ket_time'] = pd.to_datetime(ket_time_df)
ket['Admin time'] = pd.to_datetime(ket['Admin time'], errors='coerce', format='%H:%M:%S')
# set non-number values in the 'Ketanserin conc (ng/ml)' column to 0
ket['Ketanserin_conc_ng_mL'] = pd.to_numeric(ket['Ketanserin conc (ng/ml)'], errors='coerce').fillna(0)

# read the functional scans table
func_scans = pd.read_csv('data/func_scans_table_outliers.csv')

subjects = func_scans['subject'].unique()
sessions = func_scans['session'].unique()
for subject in subjects:
    subject_int = int(subject[4:])
    sessions_for_subject = func_scans[func_scans['subject'] == subject]['session'].unique()
    for ses in sessions_for_subject:
        # set session-level stuff, including drug admin time and time since drug admin for each measurement
        if ses == 'ses-KET':
            ket_subject = ket[ket['CIMBI ID'] == subject_int]
            drug_date = pd.to_datetime(ket_subject['First day of intervention'].values[0])
            drug_admin_time = pd.to_datetime(ket_subject['Admin time'].values)
            if len(np.unique(drug_admin_time)) != 1:
                raise ValueError(f"Multiple drug administration times found for subject {subject}: {drug_admin_time}")
            
            drug_admin_timestamp = drug_date.replace(hour=drug_admin_time[0].hour, minute=drug_admin_time[0].minute, second=drug_admin_time[0].second)
            ket_subject['time_since_admin'] = ((ket_subject['ket_time'] - drug_admin_timestamp).dt.total_seconds() / 60).round()
            time30ket0 = (ket_subject['time_since_admin'] > 30) & (ket_subject['Ketanserin conc (ng/ml)'] == 0)
            if time30ket0.sum() > 0:
                print(f"Warning: Subject {subject} has ketanserin measurements with time since admin > 30 minutes and value 0. These will be set to NaN since they are likely a mistake.")
                ket_subject.loc[time30ket0, 'Ketanserin_conc_ng_mL'] = np.nan
        elif ses == 'ses-PSI':
            ppl_sdi_subject = ppl_sdi[ppl_sdi['CIMBI.ID'] == subject_int]
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

        ############# loop through scans for the subject and set scan-level stuff, including appropriate drug measurements
        subject_scans = func_scans[(func_scans['subject'] == subject) & (func_scans['session'] == ses)]
        
        for idx, scan in subject_scans.iterrows():
            try:
                func_scans.at[idx, 'age'] = age_sex.loc[age_sex['Cimbi'] == subject_int, 'Age'].values[0]
                func_scans.at[idx, 'sex'] = age_sex.loc[age_sex['Cimbi'] == subject_int, 'Sex'].values[0]
            except:
                func_scans.at[idx, 'age'] = np.nan
                func_scans.at[idx, 'sex'] = np.nan

            # compute the scan mid timestamp and time since drug administration for the scan
            scan_start_timestamp = pd.to_datetime(scan['scan_start_time'])
            scan_end_timestamp = pd.to_datetime(scan['scan_end_time'])
            scan_mid_timestamp = scan_start_timestamp + (scan_end_timestamp - scan_start_timestamp) / 2
            scan_mid_timestamp = scan_mid_timestamp.replace(microsecond=0) # for brevity...

            # find matching ket and sdi and input them into the func_scans table
            if ses == 'ses-KET':
                scan_time_since_drug_admin = np.round((scan_mid_timestamp - drug_admin_timestamp).total_seconds()/60)
                func_scans.at[idx, 'scan_min_since_admin'] = scan_time_since_drug_admin
                func_scans.at[idx, 'drug_admin_time'] = drug_admin_timestamp

                if alignment_mode == 'closest_after_scan':
                    ket_time_since_scan = ((ket_subject['ket_time'] - scan_end_timestamp).dt.total_seconds()/60).round()
                    if scan_time_since_drug_admin < 0:
                        # if the scan was before the drug administration, we can only use the measurements before the drug administration
                        ket_time_since_scan = ket_time_since_scan[ket_subject['time_since_admin'] <= 0]
                    else:
                        # find the lowest positive value in the ket_time_since_scan where there is a measurement in the Ketanserin_conc_ng_mL column
                        ket_time_since_scan = ket_time_since_scan[ket_subject['Ketanserin_conc_ng_mL'].notna()]
                    closest_ket_time_since_scan = ket_time_since_scan[ket_time_since_scan > 0].min()
                    try:
                        closest_ket_argmin = ket_time_since_scan[ket_time_since_scan > 0].idxmin()
                        
                        # no rules, just take the closest measurement after the scan, even if it's far away, since ketanserin has a long half-life and measurements after the scan are more likely to reflect the ketanserin level during the scan than measurements before the scan
                        func_scans.at[idx, 'ket (ng/mL)'] = ket_subject['Ketanserin_conc_ng_mL'][closest_ket_argmin]
                        func_scans.at[idx, 'SDI'] = ket_subject['SDI'][closest_ket_argmin]
                        func_scans.at[idx, 'ket_min_since_scan'] = ket_time_since_scan[closest_ket_argmin]
                        func_scans.at[idx, 'ket_time'] = ket_subject['ket_time'][closest_ket_argmin]
                        func_scans.at[idx, 'ket_min_since_admin'] = ket_subject['time_since_admin'][closest_ket_argmin]
                    except:
                        func_scans.at[idx, 'ket (ng/mL)'] = np.nan
                        func_scans.at[idx, 'SDI'] = np.nan
                        func_scans.at[idx, 'ket_min_since_scan'] = np.nan
                        func_scans.at[idx, 'ket_time'] = pd.NaT
                        func_scans.at[idx, 'ket_min_since_admin'] = np.nan

                elif alignment_mode == 'closest_from_midscan':

                    ket_time_since_scan = ((ket_subject['ket_time'] - scan_mid_timestamp).dt.total_seconds()/60).round()

                    if scan_time_since_drug_admin < 0:
                        # if the scan was before the drug administration, we can only use the measurements before the drug administration
                        ket_time_since_scan = ket_time_since_scan[ket_subject['time_since_admin'] <= 0]
                    else:
                        # find the lowest positive value in the ket_time_since_scan where there is a measurement in the Ketanserin_conc_ng_mL column
                        ket_time_since_scan = ket_time_since_scan[ket_subject['Ketanserin_conc_ng_mL'].notna()]
                    closest_ket_time_since_scan = ket_time_since_scan.abs().min()
                    closest_ket_argmin = ket_time_since_scan.abs().idxmin()
                    if scan_time_since_drug_admin < 200:
                        closest_acceptable_time_since_scan = 40
                    else:                    
                        closest_acceptable_time_since_scan = 20

                    if np.abs(ket_time_since_scan[closest_ket_argmin]) > closest_acceptable_time_since_scan:
                        if scan_time_since_drug_admin < 0:
                            func_scans.at[idx, 'ket (ng/mL)'] = 0
                            func_scans.at[idx, 'SDI'] = 0
                            func_scans.at[idx, 'ket_min_since_scan'] = np.nan
                            func_scans.at[idx, 'ket_time'] = pd.NaT
                            func_scans.at[idx, 'ket_min_since_admin'] = np.nan
                        else:
                            # print(f"Warning: Scan {scan['task']} {scan['run']} for subject {subject} has a ketanserin measurement more than 20 minutes away from the scan. Closest measurement since scan: {ket_time_since_scan[closest_ket_argmin]} minutes.")
                            print(f"Excluding ketanserin for scan {scan['task']} {scan['run']} for subject {subject}. Closest: {ket_time_since_scan[closest_ket_argmin]} minutes.")
                            func_scans.at[idx, 'ket (ng/mL)'] = np.nan
                            func_scans.at[idx, 'SDI'] = np.nan
                            func_scans.at[idx, 'ket_min_since_scan'] = np.nan
                            func_scans.at[idx, 'ket_time'] = pd.NaT
                            func_scans.at[idx, 'ket_min_since_admin'] = np.nan
                    else:
                        # add the corresponding measurement to the scan
                        func_scans.at[idx, 'ket (ng/mL)'] = ket_subject['Ketanserin_conc_ng_mL'][closest_ket_argmin]
                        func_scans.at[idx, 'SDI'] = ket_subject['SDI'][closest_ket_argmin]
                        func_scans.at[idx, 'ket_min_since_scan'] = ket_time_since_scan[closest_ket_argmin]
                        func_scans.at[idx, 'ket_time'] = ket_subject['ket_time'][closest_ket_argmin]
                        func_scans.at[idx, 'ket_min_since_admin'] = ket_subject['time_since_admin'][closest_ket_argmin]

            elif ses == 'ses-PSI':
                # count minutes after drug administration
                scan_time_since_drug_admin = np.round((scan_mid_timestamp - drug_admin_timestamp).total_seconds()/60)
                func_scans.at[idx, 'scan_min_since_admin'] = scan_time_since_drug_admin
                func_scans.at[idx, 'drug_admin_time'] = drug_admin_timestamp

                if scan_time_since_drug_admin < 0 and scan['subject']=='sub-55772':
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

                if alignment_mode == 'closest_after_scan':
                    ppl_time_since_scan = ((ppl_sdi_subject['ppl_time'] - scan_end_timestamp).dt.total_seconds()/60).round()
                    sdi_time_since_scan = ((ppl_sdi_subject['sdi_time'] - scan_end_timestamp).dt.total_seconds()/60).round()
                    if scan_time_since_drug_admin < 0:
                        # if the scan was before the drug administration, we can only use the measurements before the drug administration
                        ppl_time_since_scan = ppl_time_since_scan[ppl_sdi_subject['ppl_time_since_admin'] < 0]
                        sdi_time_since_scan = sdi_time_since_scan[ppl_sdi_subject['sdi_time_since_admin'] < 0]
                    else:
                        # find the lowest positive value in the ppl_meas_time_since_scan where there is a measurement in the psi_conc_mcg_per_L column
                        ppl_time_since_scan = ppl_time_since_scan[ppl_sdi_subject['psi_conc_mcg_per_L'].notna()]
                        sdi_time_since_scan = sdi_time_since_scan[ppl_sdi_subject['SDI_score'].notna()]
                    closest_ppl_time_since_scan = ppl_time_since_scan[ppl_time_since_scan > 0].min()
                    try:
                        closest_ppl_argmin = ppl_time_since_scan[ppl_time_since_scan > 0].idxmin()
                        closest_sdi_time_since_scan = sdi_time_since_scan[sdi_time_since_scan > 0].min()
                        closest_sdi_argmin = sdi_time_since_scan[sdi_time_since_scan > 0].idxmin()

                        func_scans.at[idx, 'PPL_mcg/L'] = ppl_sdi_subject['psi_conc_mcg_per_L'][closest_ppl_argmin]
                        func_scans.at[idx, 'ppl_min_since_scan'] = ppl_time_since_scan[closest_ppl_argmin]
                        func_scans.at[idx, 'ppl_min_since_admin'] = ppl_sdi_subject['ppl_time_since_admin'][closest_ppl_argmin]
                        func_scans.at[idx, 'ppl_time'] = ppl_sdi_subject['ppl_time'][closest_ppl_argmin]
                        func_scans.at[idx, 'SDI'] = ppl_sdi_subject['SDI_score'][closest_sdi_argmin]
                        func_scans.at[idx, 'sdi_min_since_scan'] = sdi_time_since_scan[closest_sdi_argmin]
                        func_scans.at[idx, 'sdi_min_since_admin'] = ppl_sdi_subject['sdi_time_since_admin'][closest_sdi_argmin]
                        func_scans.at[idx, 'sdi_time'] = ppl_sdi_subject['sdi_time'][closest_sdi_argmin]
                    except:
                        func_scans.at[idx, 'PPL_mcg/L'] = np.nan
                        func_scans.at[idx, 'ppl_min_since_scan'] = np.nan
                        func_scans.at[idx, 'ppl_min_since_admin'] = np.nan
                        func_scans.at[idx, 'ppl_time'] = pd.NaT
                        func_scans.at[idx, 'SDI'] = np.nan
                        func_scans.at[idx, 'sdi_min_since_scan'] = np.nan
                        func_scans.at[idx, 'sdi_min_since_admin'] = np.nan
                        func_scans.at[idx, 'sdi_time'] = pd.NaT

                elif alignment_mode == 'closest_from_midscan':

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

                    if scan_time_since_drug_admin > 200:
                        closest_acceptable_time_since_scan = 40
                    else:                    
                        closest_acceptable_time_since_scan = 20

                    # if scan_time_since_drug_admin > 0:# and scan_time_since_drug_admin < 200:
                    if np.abs(ppl_time_since_scan[closest_ppl_argmin]) > closest_acceptable_time_since_scan:
                        # print(f"Warning: Scan {scan['run']} for subject {subject} has a PPL measurement more than 20 minutes away from the scan time. Closest PPL measurement since scan: {ppl_time_since_scan[closest_ppl_argmin]} minutes.")
                        if subject in ['sub-55992','sub-57142'] and np.abs(ppl_time_since_scan[closest_ppl_argmin]) < 30:
                            print(f"Accepting ppl for scan {scan['task']} {scan['run']} for subject {subject} despite the warning, since the ppl trajectory matches the other measurements.")
                        elif scan_time_since_drug_admin < 0:
                            func_scans.at[idx, 'PPL_mcg/L'] = 0
                            func_scans.at[idx, 'ppl_min_since_scan'] = np.nan
                            func_scans.at[idx, 'ppl_min_since_admin'] = np.nan
                            func_scans.at[idx, 'ppl_time'] = pd.NaT
                        else:
                            print(f"Excluding ppl for scan {scan['task']} {scan['run']} for subject {subject}. Closest: {ppl_time_since_scan[closest_ppl_argmin]} minutes.")
                            func_scans.at[idx, 'PPL_mcg/L'] = np.nan
                            func_scans.at[idx, 'ppl_min_since_scan'] = np.nan
                            func_scans.at[idx, 'ppl_min_since_admin'] = np.nan
                            func_scans.at[idx, 'ppl_time'] = pd.NaT
                    else:
                        func_scans.at[idx, 'PPL_mcg/L'] = ppl_sdi_subject['psi_conc_mcg_per_L'][closest_ppl_argmin]
                        func_scans.at[idx, 'ppl_min_since_scan'] = ppl_time_since_scan[closest_ppl_argmin]
                        func_scans.at[idx, 'ppl_min_since_admin'] = ppl_sdi_subject['ppl_time_since_admin'][closest_ppl_argmin]
                        func_scans.at[idx, 'ppl_time'] = ppl_sdi_subject['ppl_time'][closest_ppl_argmin]

                    if np.abs(sdi_time_since_scan[closest_sdi_argmin]) > closest_acceptable_time_since_scan:
                        # print(f"Warning: Scan {scan['run']} for subject {subject} has a SDI measurement more than 20 minutes away from the scan time. Closest SDI measurement since scan: {sdi_time_since_scan[closest_sdi_argmin]} minutes.")
                        if subject in ['sub-55992','sub-56145','sub-57142','sub-57193'] and np.abs(sdi_time_since_scan[closest_sdi_argmin]) < 30:
                            # these subjects are accepted anyways
                            print(f"Accepting sdi for scan {scan['task']} {scan['run']} for subject {subject} despite the warning, since the sdi trajectory matches the other measurements.")
                        elif scan_time_since_drug_admin < 0:
                            func_scans.at[idx, 'SDI'] = 0
                            func_scans.at[idx, 'sdi_min_since_scan'] = np.nan
                            func_scans.at[idx, 'sdi_min_since_admin'] = np.nan
                            func_scans.at[idx, 'sdi_time'] = pd.NaT
                        else:
                            print(f"Excluding sdi for scan {scan['task']} {scan['run']} for subject {subject}. Closest: {sdi_time_since_scan[closest_sdi_argmin]} minutes.")
                            func_scans.at[idx, 'SDI'] = np.nan
                            func_scans.at[idx, 'sdi_min_since_scan'] = np.nan
                            func_scans.at[idx, 'sdi_min_since_admin'] = np.nan
                            func_scans.at[idx, 'sdi_time'] = pd.NaT
                    else:
                        func_scans.at[idx, 'SDI'] = SDI
                        func_scans.at[idx, 'sdi_min_since_scan'] = sdi_time_since_scan[closest_sdi_argmin]
                        func_scans.at[idx, 'sdi_min_since_admin'] = ppl_sdi_subject['sdi_time_since_admin'][closest_sdi_argmin]
                        func_scans.at[idx, 'sdi_time'] = ppl_sdi_subject['sdi_time'][closest_sdi_argmin]
            if ses in ['ses-KET','ses-PSI'] and not np.isnan(scan_time_since_drug_admin):
                for time_interval in config['time_intervals']:
                    if scan_time_since_drug_admin>config['time_intervals'][time_interval][0] and scan_time_since_drug_admin<config['time_intervals'][time_interval][1]:
                        func_scans.at[idx, 'time_interval'] = time_interval
                        break
            
            asl_scans = glob('data/raw/'+subject+'/'+ses+'/perf/sub-*_ses-*_asl.json')
            asl_acq_time = []
            asl_run = []
            asl_generatedfrom = []
            for asl_scan in asl_scans:
                with open(asl_scan, 'r') as f:
                    cbf_scan_json = json.load(f)
                asl_acq_time.append(cbf_scan_json['AcquisitionTime'])
                asl_generatedfrom.append(cbf_scan_json['generatedfrom'])
                # asl_run.append(asl_scan[asl_scan.find('run-')+4:asl_scan.find('_space-MNI152NLin6Asym_res-2_desc-preproc_asl.json')])
                asl_run.append(asl_scan)
            date = scan.scan_start_time[:10]
            if len(asl_acq_time)>0:
                asl_acq_time = pd.to_datetime(date + ' ' + pd.Series(asl_acq_time))
                # remove milliseconds
                asl_acq_time = asl_acq_time.dt.floor('s')

                if ses in ['ses-KET','ses-PSI']:
                    if scan_time_since_drug_admin < 0:
                        asl_run = np.array(asl_run)[asl_acq_time < drug_admin_timestamp]
                        asl_acq_time = asl_acq_time[asl_acq_time < drug_admin_timestamp]

                        scan_time_since_asl = (asl_acq_time - scan_mid_timestamp).dt.total_seconds()/60
                        closest_asl_time_since_scan = np.abs(scan_time_since_asl).min()
                        closest_asl_argmin = np.abs(scan_time_since_asl).argmin()
                        func_scans.at[idx, 'asl_filename'] = asl_scans[closest_asl_argmin].replace('space-MNI152NLin6Asym_res-2_desc-preproc_asl.json', 'atlas-4S256Parcels_cbf.tsv')
                        func_scans.at[idx, 'asl_min_since_scan'] = scan_time_since_asl[closest_asl_argmin].round()
                        func_scans.at[idx, 'asl_start_time'] = asl_acq_time[closest_asl_argmin]
                        func_scans.at[idx, 'asl_min_since_admin'] = np.round((asl_acq_time[closest_asl_argmin] - drug_admin_timestamp).total_seconds()/60)
                        func_scans.at[idx, 'asl_generatedfrom'] = asl_generatedfrom[closest_asl_argmin]
                        continue
                    else:
                        scan_time_since_asl = (asl_acq_time - scan_mid_timestamp).dt.total_seconds()/60
                        closest_asl_time_since_scan = np.abs(scan_time_since_asl).min()
                        closest_asl_argmin = np.abs(scan_time_since_asl).argmin()
                        
                        if closest_asl_time_since_scan < 30:
                            func_scans.at[idx, 'asl_filename'] = asl_scans[closest_asl_argmin].replace('space-MNI152NLin6Asym_res-2_desc-preproc_asl.json', 'atlas-4S256Parcels_cbf.tsv')
                            func_scans.at[idx, 'asl_min_since_scan'] = scan_time_since_asl[closest_asl_argmin].round()
                            func_scans.at[idx, 'asl_start_time'] = asl_acq_time[closest_asl_argmin]
                            func_scans.at[idx, 'asl_min_since_admin'] = np.round((asl_acq_time[closest_asl_argmin] - drug_admin_timestamp).total_seconds()/60)
                            func_scans.at[idx, 'asl_generatedfrom'] = asl_generatedfrom[closest_asl_argmin]
                        else:
                            func_scans.at[idx, 'asl_filename'] = np.nan
                            func_scans.at[idx, 'asl_min_since_scan'] = np.nan
                            func_scans.at[idx, 'asl_start_time'] = pd.NaT
                            func_scans.at[idx, 'asl_min_since_admin'] = np.nan
                            func_scans.at[idx, 'asl_generatedfrom'] = np.nan
                else:
                    scan_time_since_asl = (asl_acq_time - scan_mid_timestamp).dt.total_seconds()/60
                    closest_asl_time_since_scan = np.abs(scan_time_since_asl).min()
                    closest_asl_argmin = np.abs(scan_time_since_asl).argmin()
                    func_scans.at[idx, 'asl_filename'] = asl_scans[closest_asl_argmin].replace('space-MNI152NLin6Asym_res-2_desc-preproc_asl.json', 'atlas-4S256Parcels_cbf.tsv')
                    func_scans.at[idx, 'asl_min_since_scan'] = scan_time_since_asl[closest_asl_argmin].round()
                    func_scans.at[idx, 'asl_start_time'] = asl_acq_time[closest_asl_argmin]
                    func_scans.at[idx, 'asl_min_since_admin'] = np.nan
                    func_scans.at[idx, 'asl_generatedfrom'] = asl_generatedfrom[closest_asl_argmin]
            else:
                func_scans.at[idx, 'asl_filename'] = np.nan
                func_scans.at[idx, 'asl_min_since_scan'] = np.nan
                func_scans.at[idx, 'asl_start_time'] = pd.NaT
                func_scans.at[idx, 'asl_min_since_admin'] = np.nan
                func_scans.at[idx, 'asl_generatedfrom'] = np.nan
        
        
func_scans['include_manual_qc'] = func_scans['include_manual_qc'].astype('boolean')
func_scans.loc[func_scans['session'] != 'ses-PSI', 'include_manual_qc'] = pd.NA
func_scans['include_scan_coil_numvols'] = func_scans['include_scan_coil_numvols'].astype('boolean')
func_scans.loc[func_scans['session'] != 'ses-PSI', 'include_scan_coil_numvols'] = pd.NA
    

# save 1: only ses-PSI but include filenames
func_scans1 = func_scans[['subject', 'raw_id', 'session', 'task', 'run','age','sex',
                         'drug_admin_time','ppl_time','sdi_time','asl_start_time',
                         'scan_start_time','scan_end_time', 'scan_min_since_admin', 'ppl_min_since_admin','sdi_min_since_admin','asl_min_since_admin',
                        'ppl_min_since_scan','sdi_min_since_scan','asl_min_since_scan','PPL_mcg/L', 'SDI', 'time_interval',
                        'tr', 'te','num_vols','ped', 'coil_name', 'coil_active', 'scanner', 'include_scan_coil_numvols','include_manual_qc',
                        'ratio_outliers_fd'+str(0.5)+'_std_dvars'+str(1000),'mean_fd', 'mean_std_dvars','max_fd','outlier_locs',
                        'scan_filename','preproc_filename_volumetric', 'preproc_filename_cifti','preproc_filename_cifti_despiked','preproc_filename_cifti_aroma','asl_filename','asl_generatedfrom']]
func_scans1 = func_scans1[func_scans1['session'] == 'ses-PSI']
func_scans1_rest = func_scans1[func_scans1['task'] == 'task-rest']
func_scans1.to_csv(f'data/func_scans_table_outliers_ses-PSI_PPLSDI_'+alignment_mode+'.csv', index=False)
func_scans1_rest.to_csv(f'data/func_scans_table_outliers_ses-PSI_task-rest_PPLSDI_'+alignment_mode+'.csv', index=False)

# save 2: all sessions and tasks but without filenames
func_scans['ratio_outliers_fd'+str(0.5)] = func_scans['ratio_outliers_fd'+str(0.5)+'_std_dvars'+str(1000)]
try:
    func_scans['asl_filename'] = func_scans['asl_filename'].str.replace('data/preprocessed_asl/', 'cbf/')
except:
    pass
func_scans2 = func_scans[['subject', 'raw_id', 'session', 'task', 'run','age','sex',
                         'drug_admin_time','ppl_time','sdi_time','ket_time','asl_start_time',
                         'scan_start_time','scan_end_time', 'scan_min_since_admin', 'ppl_min_since_admin','sdi_min_since_admin','ket_min_since_admin','asl_min_since_admin',
                        'ppl_min_since_scan','sdi_min_since_scan','ket_min_since_scan','asl_min_since_scan','PPL_mcg/L', 'SDI', 'ket (ng/mL)','time_interval',
                        'tr', 'te','num_vols','ped', 'coil_name', 'coil_active', 'scanner', 'include_scan_coil_numvols','include_manual_qc',
                        'ratio_outliers_fd'+str(0.5),'mean_fd', 'max_fd','outlier_locs',
                        'asl_filename']]
func_scans2.to_csv('data/func_scans_table_outliers_PPLSDIket_'+alignment_mode+'.csv', index=False)
func_scans2.to_csv('/mrdata/np2/p3/denoised_parcellated_np2p3/func_scans_table_outliers_PPLSDIket_'+alignment_mode+'.csv', index=False)
        
        
    
