from glob import glob
import os
import pandas as pd
import json
import re
import nibabel as nib
import pydicom

def make_table_of_func_scans(scan_dir, derivatives_dir):
    """
    Create a table of scans from the specified directory.

    Parameters:
    scan_dir (str): The directory containing scan files.

    Returns:
    pd.DataFrame: A DataFrame containing the scan file names and their paths.
    """

    subjects = glob(os.path.join(derivatives_dir, 'sub-*'))
    if not subjects:
        raise ValueError(f"No subjects found in the directory: {scan_dir}") 
    
    subjects.sort()  # sort subjects for consistent order
    # keep only directories, not files
    subjects = [s for s in subjects if os.path.isdir(s)]
    # ppl_sdi = pd.read_excel('data/SDI_PPL_P3_20220202.xlsx')
    mrids = pd.read_csv('data/raw/code/spreadsheets/NeuroPharm2-P3-MR_bidsdata_path.csv')
    scans_collector = []
    for subject in subjects:
        print(f"Processing subject: {os.path.basename(subject)}")
        # ppl_sdi_subject = ppl_sdi[ppl_sdi['CIMBI.ID'] == int(os.path.basename(subject)[4:])]
        # if int(os.path.basename(subject)[4:]) in [56133, 56314, 56729, 57219]:
        #     date = '1999-12-31'  # missing date, set to a dummy date
        # else:
        #     date = pd.to_datetime(ppl_sdi_subject['MR_scan_date']).iloc[0].date().isoformat()
        sessions = glob(os.path.join(subject, 'ses-*'))
        sessions.sort()  # sort sessions for consistent order
        if not sessions:
            print(f"No sessions found for subject: {subject}")
            # raise ValueError(f"No sessions found for subject: {subject}")
        for session in sessions:
            scans = glob(os.path.join(session, 'func/*desc-preproc_bold.nii.gz'))
            # scans.sort()
            scans = sorted(scans, key=lambda p: (re.search(r'task-([^_]+).*_run-(\d+)', p).group(1), int(re.search(r'run-(\d+)', p).group(1))))
            if not scans:
                print(f"No scans found for session: {session}")

            # find .tsv-file
            # scan_tsv_files = glob(os.path.join(session, '*scans.tsv'))
            # if not scan_tsv_files:
            #     print(f"No scans.tsv file found for session: {session}")
            #     continue
            # scan_tsv = pd.read_csv(scan_tsv_files[0], sep='\t')
            for scan in scans:
                # extract data from the scan file name
                # extract the task from the scan file name as where it begins with 'task-' and ends with '_'
                task = 'task-' + os.path.basename(scan).split('task-')[1].split('_acq')[0]
                run = 'run-' + os.path.basename(scan).split('run-')[1].split('_')[0]
                ped = 'dir-' + os.path.basename(scan).split('dir-')[1].split('_')[0]
                cimbi = int(os.path.basename(subject)[4:])
                
                json_file_preproc = scan.replace('_bold.nii.gz', '_bold.json')
                with open(json_file_preproc, 'r') as f:
                    metadata_preproc = json.load(f)
                sources = metadata_preproc.get('Sources', 'N/A')
                # find the source that starts with 'bids:raw:'
                raw_source = [s for s in sources if s.startswith('bids:raw:')]
                if len(raw_source) != 1:
                    print(f"Warning: Scan {os.path.basename(scan)} has unexpected number of raw sources: {sources}")
                    continue
                raw_source_file = scan_dir+raw_source[0].replace('bids:raw:', '')
                if 'acq-MB8ep2d' in raw_source_file:
                    raw_source_file = raw_source_file.replace('acq-MB8ep2d', 'acq-mb')

                if not os.path.exists(raw_source_file):
                    raise ValueError(f"Warning: Raw source file {raw_source_file} does not exist for scan {os.path.basename(scan)}")
                    
                json_file_raw = raw_source_file.replace('.nii.gz', '.json')
                with open(json_file_raw, 'r') as f:
                    metadata_raw = json.load(f)
                
                generatedfrom = metadata_raw.get('generatedfrom', 'N/A')
                mrids_subses = mrids[(mrids['CIMBI'] == cimbi) & (mrids['source'].apply(lambda x: os.path.basename(x[:-5])) == os.path.basename(session))]
                ses_ids = mrids_subses[['MRid_1', 'MRid_2', 'MRid_3']].values.flatten()
                ses_ids = [x for x in ses_ids if pd.notna(x)]

                if os.path.basename(subject)=='sub-56118' and os.path.basename(session)=='ses-KET' and task=='task-rest' and run=='run-1':
                    ses_ids = [ses_ids[0]]
                elif os.path.basename(subject)=='sub-56118' and os.path.basename(session)=='ses-KET' and task=='task-rest' and run=='run-2':
                    ses_ids = [ses_ids[1]]
                elif os.path.basename(subject)=='sub-56017' and os.path.basename(session)=='ses-BL' and task=='task-aarhusmusic' and run=='run-1':
                    ses_ids = ['n0028'] # not n0027 for some reason...
                    
                already_used = False
                date = None
                for raw_id in ses_ids:
                    if 'p' in raw_id:
                        # find the path
                        rawpath = glob('/rawdata/mr-rh/MRraw/prisma/'+raw_id+'*')[0]
                    elif 'n' in raw_id:
                        rawpath = glob('/rawdata/mr-rh/MRraw/mr001/'+raw_id+'*')[0]
                    # check if generatedfrom is a subfolder of 'path'
                    folders = glob(os.path.join(rawpath, '*'))
                    if any(generatedfrom in folder for folder in folders):
                        if already_used:
                            raise ValueError(f"Warning: Scan {os.path.basename(scan)} has multiple matching raw sources: {sources}")
                        match = rawpath+'/'+generatedfrom
                        anyfile = glob(match+'/*')
                        dicom = pydicom.dcmread(anyfile[0], stop_before_pixels=True)
                        try:
                            date = dicom.AcquisitionDate
                        except:
                            date = dicom.StudyDate
                        already_used = True
                        raw_id_used = raw_id
                if date is None:
                    print(f"Warning: Scan {os.path.basename(scan)} has no matching raw source found in the spreadsheet")
                    continue

                acquisition_time = metadata_raw.get('AcquisitionTime', 'N/A')
                
                # if os.path.basename(subject) == 'sub-56145' and os.path.basename(session) == 'ses-PSI':
                #     # set the date to october 30th 2018
                #     if pd.to_datetime(acquisition_time).hour < 8:
                #         date = '2018-10-31'
                #     else:
                #         date = '2018-10-30'
                # else:
                    # date = scan_row['acq_time'].item()[:10]
                timestamp = pd.to_datetime(date + ' ' + acquisition_time)
                # acquisition_time = datetime.strptime(acquisition_time, "%H:%M:%S.%f").time()
                tr = metadata_raw.get('RepetitionTime', 'N/A')
                te = metadata_raw.get('EchoTime', 'N/A')
                # ped = metadata_raw.get('PhaseEncodingDirection', 'N/A')
                coil_name = metadata_raw.get('ReceiveCoilName', 'N/A')
                coil_active = metadata_raw.get('ReceiveCoilActiveElements', 'N/A')
                scanner = metadata_raw.get('ManufacturersModelName', 'N/A')
                if scanner == 'Prisma':
                    scanner = 'MR001'
                elif scanner == 'Prisma_fit':
                    scanner = 'MR45'
                
                preproc_scan_cifti = scan.replace('_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz', '_space-fsLR_den-91k_bold.dtseries.nii')
                preproc_scan_cifti_despiked = scan.replace('_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz', '_space-fsLR_den-91k_desc-despiked_bold.dtseries.nii')
                preproc_scan_cifti_aroma = scan.replace('_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz', '_space-fsLR_den-91k_desc-smoothAROMAnonaggr_bold.dtseries.nii')
                if not os.path.exists(preproc_scan_cifti):
                    preproc_scan_cifti = None
                if not os.path.exists(preproc_scan_cifti_despiked):
                    preproc_scan_cifti_despiked = None
                if not os.path.exists(preproc_scan_cifti_aroma):
                    preproc_scan_cifti_aroma = None

                #load the nifti file to get the header information
                if preproc_scan_cifti is not None:
                    cifti_img = nib.load(preproc_scan_cifti)
                    num_volumes = cifti_img.shape[0]
                else:
                    nifti_img = nib.load(scan)
                    # get the number of volumes (timepoints) in the nifti file
                    num_volumes = nifti_img.shape[3]

                scan_length_seconds = num_volumes * tr
                scan_end_time = timestamp + pd.Timedelta(seconds=scan_length_seconds)

                include_scan = True
                if scanner=='MR001':
                    if coil_active not in ['HEA;HEP']:
                        print(f"Scan {os.path.basename(subject)} {session} {task} {run} had unexpected coil active elements: {coil_active}")
                        include_scan = False
                    if task == 'task-rest':
                        if num_volumes not in [300,375,750]:
                            print(f"Scan {os.path.basename(subject)} {session} {task} {run} had unexpected number of volumes: {num_volumes}")
                            include_scan = False
                elif scanner=='MR45':
                    if coil_active not in ['HC1-6']:
                        print(f"Scan {os.path.basename(subject)} {session} {task} {run} had unexpected coil active elements: {coil_active}")
                        include_scan = False
                    if task == 'task-rest':
                        if num_volumes not in [300]:
                            print(f"Scan {os.path.basename(subject)} {session} {task} {run} had unexpected number of volumes: {num_volumes}")
                            include_scan = False

                scans_collector.append({
                    'subject': os.path.basename(subject),
                    'raw_id': raw_id_used,
                    'session': os.path.basename(session),
                    'task': task,
                    'run': run,
                    'scan_start_time': timestamp.replace(microsecond=0),
                    'scan_end_time': scan_end_time.replace(microsecond=0),  
                    'tr': tr,
                    'te': te,
                    'num_vols': num_volumes,
                    'ped': ped,
                    'coil_name': coil_name,
                    'coil_active': coil_active,
                    'scanner': scanner,
                    'include_scan_coil_numvols': include_scan,
                    'scan_filename': raw_source_file,
                    'preproc_filename_volumetric': scan,
                    'preproc_filename_cifti': preproc_scan_cifti,
                    'preproc_filename_cifti_despiked': preproc_scan_cifti_despiked,
                    'preproc_filename_cifti_aroma': preproc_scan_cifti_aroma
                })
                
    # Create a DataFrame from the scans list
    df = pd.DataFrame(scans_collector)

    # manual discarding of some scans, only those that are not just ordinary motion
    df['include_manual_qc'] = True

    # set all scans without ses-PSI to include_manual_qc False, as we only want to include scans from ses-PSI in the spectral analysis
    # df.loc[df['session'] != 'ses-PSI', 'include_manual_qc'] = None

    # three scans for sub-55746 have some weird oscillatory noise in the data at some sections
    df.loc[(df['subject'] == 'sub-55746') & (df['session'] == 'ses-PSI') & (df['task'] == 'task-rest') & (df['run'] == 'run-1'), 'include_manual_qc'] = False
    df.loc[(df['subject'] == 'sub-55746') & (df['session'] == 'ses-PSI') & (df['task'] == 'task-rest') & (df['run'] == 'run-2'), 'include_manual_qc'] = False
    df.loc[(df['subject'] == 'sub-55746') & (df['session'] == 'ses-PSI') & (df['task'] == 'task-rest') & (df['run'] == 'run-5'), 'include_manual_qc'] = False
    
    # bad registration
    df.loc[(df['subject'] == 'sub-56145') & (df['session'] == 'ses-PSI') & (df['task'] == 'task-rest') & (df['run'] == 'run-4'), 'include_manual_qc'] = False
    df.loc[(df['subject'] == 'sub-56165') & (df['session'] == 'ses-PSI') & (df['task'] == 'task-rest') & (df['run'] == 'run-2'), 'include_manual_qc'] = False
    df.loc[(df['subject'] == 'sub-56165') & (df['session'] == 'ses-PSI') & (df['task'] == 'task-rest') & (df['run'] == 'run-3'), 'include_manual_qc'] = False
    df.loc[(df['subject'] == 'sub-56165') & (df['session'] == 'ses-PSI') & (df['task'] == 'task-rest') & (df['run'] == 'run-4'), 'include_manual_qc'] = False
    df.loc[(df['subject'] == 'sub-55809') & (df['session'] == 'ses-PSI') & (df['task'] == 'task-rest') & (df['run'] == 'run-3'), 'include_manual_qc'] = False
    
    # bad fieldmap correction, check later if rerun of fmriprep without fieldmaps helps, rerun started 22/10/2025
    df.loc[(df['subject'] == 'sub-57140') & (df['session'] == 'ses-PSI'), 'include_manual_qc'] = False
    return df

if __name__ == "__main__":
    import os

    # delete existing files to avoid appending to them
    existing_files = glob('data/func_scans*.csv')
    for file in existing_files:
        os.remove(file)

    raw_scan_directory = 'data/raw/'
    derivatives_directory = 'data/preprocessed/'
    scans_table = make_table_of_func_scans(raw_scan_directory, derivatives_directory)
    
    # Save the DataFrame to a CSV file
    output_file = 'data/func_scans_table.csv'
    scans_table.to_csv(output_file, index=False)

    print(f"Table of functional scans saved")