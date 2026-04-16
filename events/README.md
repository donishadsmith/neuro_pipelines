# Creating Event Files

## Command Line Interface (CLI)

### Basic Usage

```bash
python bids_events_cli.py --src_dir /path/to/log/files --temp_dir /path/to/temp/dir --dst_dir /path/to/output/dir --task nback
```

### Restricting to Specific Subjects

```bash
python bids_events_cli.py --src_dir /path/to/log/files --temp_dir /path/to/temp/dir --dst_dir /path/to/output/dir --task nback --subjects 101 102
```

### Using a Subjects Visits File

When sessions are missing, a subjects visits file ensures that dates are mapped to the correct session IDs. Pass the file with `--subjects_visits_file` and specify the date format with `--subjects_visits_date_fmt`:

```bash
python bids_events_cli.py --src_dir /path/to/log/files --temp_dir /path/to/temp/dir --dst_dir /path/to/output/dir --task nback --subjects 101 102 --subjects_visits_file /path/to/visits.csv --subjects_visits_date_fmt %m/%d/%Y
```

The file must contain `participant_id` and `date` columns. List all visit dates in chronological order per subject and use `NaN` for missing sessions:

| participant_id | date       |
|----------------|------------|
| 101            | 01/02/2022 |
| 101            | NaN        |
| 101            | 03/02/2022 |
| 102            | 01/02/2024 |

### Full Help

```bash
python bids_events_cli.py --help
```

## Streamlit Graphical User Interface (GUI)

For a point-and-click interface, launch the Streamlit app:

```bash
streamlit run bids_events_app.py
```

