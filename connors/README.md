# Conners 4 Score Extraction

Extracts T-scores and percentiles from Conners 4 PDF reports and saves them to a CSV file. PDFs are first renamed to a standardized format (``sub-{id}_date-{date}_rater-{rater}.pdf``) and sorted into ``child/`` and ``parent/`` subdirectories under a ``reformatted_filenames/`` folder.

**Extracted scores include:** Inattention/Executive Dysfunction, Hyperactivity, Impulsivity, Emotional Dysregulation, Depressed Mood, Anxious Thoughts, Schoolwork, Peer Interactions, Family Life, ADHD Inattentive, ADHD Hyperactive/Impulsive, Total ADHD Symptoms, ODD, Conduct Disorder, and ADHD Index.

## Command Line Interface (CLI)

### Basic Usage

```bash
python connors_cli.py --pdf_dir /path/to/pdf/files
```

Output CSV is saved to ``reformatted_filenames/conners_data.csv`` inside the PDF directory by default.

### Appending to an Existing CSV

```bash
python connors_cli.py --pdf_dir /path/to/pdf/files --csv_file_path /path/to/existing/conners_data.csv
```

### Restricting to Specific Subjects

```bash
python connors_cli.py --pdf_dir /path/to/pdf/files --subjects 101 102
```

### Full Help

```bash
python connors_cli.py --help
```

## Streamlit Graphical User Interface (GUI)

```bash
streamlit run connors_app.py
```
