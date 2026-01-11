# Overview

Pipeline for converting non-BIDS data to BIDS data. Assumes non-BIDS data has folder names in the form of ``"{subjectID}_{scan_date}"`` (also compatable with ``"{subjectID}_{other_content}_{scan_date}"`` as long as subject ID is first and scan date is last). Also tailored for specific tasks and scan protocols. Also assumes one run per task.

**Notes:**
- Ensure scan dates are standardized since they are sorted.
- If conversion done on Windows, may need to convert line endings from [DOS to UNIX](https://blog.bachi.net/?p=1715) for text files ``sed -i "s/\r//" filename``

Pipeline:

- Collects files ending in ".nii" and ".nii.gz"
- Compresses ".nii" files to "nii.gz" files
- Identifies the identity of the NIfTI image by checking if it is 3D or 4D and by using heuristic of "x task has n number of volumes"
- Converts data to BIDS format
- Creates a minimal metadata JSON sidecar
- Optionally create the participants TSV, dataset description JSON, and a sessions TSV.
    - If participants TSV is detected, will instead append new subjects to the dataframe.

# Usage
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

To run on HPC with SLURM:

```bash
sbatch --array=0-1 main.sb 100 101
sbatch participants_tsv.sb
```