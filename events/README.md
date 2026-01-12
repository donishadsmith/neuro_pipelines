# Creating Events

To create event file, in your create event files, run the following in your preferred terminal:

```bash
pip install nifti2bids[all]
```

```bash
python create_event_files.py --src_dir /path/to/src/dir --temp_dir /path/to/temp/dir --bids_dir /path/to/bids/dir --delete_temp_dir True --task nback 
```

To run with specific subjects:

```bash
python create_event_files.py --src_dir /path/to/src/dir --temp_dir /path/to/temp/dir --bids_dir /path/to/bids/dir --delete_temp_dir True --task nback --subjects 101 102
```

For help:

```bash
python create_event_files.py --help
```

## Subjects Visits File
When entire sessions are missing, to ensure the dates are mapped to the correct session ID, pass ``--subjects_sessions_file``:

```bash
python create_event_files.py --src_dir /path/to/src/dir --temp_dir /path/to/temp/dir --bids_dir /path/to/bids/dir --delete_temp_dir True --task nback --subjects 101 102 --subjects_visits_date_fmt %m/%d/%Y --src_data_date_fmt %Y%m%d
```

Ensure each partcipant ID in the file has all of their dates or NaN for missing dates:

| subject | scan_date  |
|---------|------------|
| 101     | 01/02/2022 |
| 101     | NaN        |
| 101     | 03/02/2022 |
| 102     | 01/02/2024 |