Neuroimaging pipelines tailored for specific tasks and scan protocols.

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

For fMRIPrep, first level, and extract contrasts, run on HPC with SLURM:

Note: "fmriprep.sb" needs an Apptainer/Singularity image of fMRIPrep and an initialized Templateflow directory. Both "first_level.sb" and "extract_contrasts.sb" requires an Apptainer/Singularity image of AFNI with R.

```bash
sbatch --array=0-20 fmriprep.sb # Runs first 21 subjects in the participants.tsv in parallel
sbatch --array=0-30 first_level.sb # Runs first 31 subjects in the participants.tsv in parallel
sbatch --array=0-10 extract_contrasts.sb # Runs first 11 subjects in the participants.tsv in parallel
```

For specific subjects:

```bash
sbatch --array=0-1 fmriprep.sb 101 110 # Runs these two subjects in parallel
sbatch --array=0-1 first_level.sb 101 102 # Runs these two subjects in parallel
sbatch --array=0-1 extract_contrasts.sb 101 102 # Runs these two subjects in parallel
```