Pipeline for converting non-BIDS data to BIDS data. Assumes non-BIDS data has folder names in the form of ``"{subjectID}_{scandate}"`` (also compatable with ``"{subjectID}_{other_content}_{scandate}"`` as long as subject ID is first and scan date is last). Also tailored for specific tasks and scan protocols.

To run pipeline, in your preferred terminal, run:

```bash
pip install nifti2bids[all]
```

```bash
python main.py --src_dir /path/to/src/dir --temp_dir /path/to/temp/dir --bids_dir /path/to/bids/dir --delete_temp_dir True --create_dataset_metadata True --add_sessions_tsv True 
```

To run only specific subjects:

```bash
python main.py --src_dir /path/to/src/dir --temp_dir /path/to/temp/dir --bids_dir /path/to/bids/dir --delete_temp_dir True --create_dataset_metadata True --add_sessions_tsv True --subjects 100 101
```

For help:

```bash
python main.py --help
```