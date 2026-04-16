# Move Files

Moves BIDS-compliant event and sessions files from a source directory into the appropriate subject subdirectories within an existing BIDS directory. Files are placed based on their BIDS entities (``sub-``, ``ses-``). Sessions TSV files go at the session level and all other files go into ``func/``.

**Notes:**
- Source files must follow BIDS naming conventions (i.e., filenames must start with ``sub-``).
- Destination subdirectories must already exist - files are skipped if the target directory is not found.
- If a file with the same name already exists in the destination, it is overwritten.

## Command Line Interface (CLI)

```bash
python move_files_cli.py --src_dir /path/to/source --bids_dir /path/to/bids
```

### Full Help

```bash
python move_files_cli.py --help
```

### Running on HPC with SLURM

```bash
sbatch move_files.sb
```

## Streamlit Graphical User Interface (GUI)

```bash
streamlit run move_files_app.py
```