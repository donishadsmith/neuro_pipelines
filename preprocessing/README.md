# Preprocessing with fMRIPrep
For fMRIPrep, first level, and extract contrasts, run on HPC with SLURM:

**Notes:**
- "fmriprep.sb" needs an Apptainer/Singularity image of fMRIPrep and an initialized Templateflow directory.
- May need to convert line endings from [DOS to UNIX](https://blog.bachi.net/?p=1715) for text files ``sed -i "s/\r//" filename``

```bash
sbatch --array=0-20 fmriprep.sb # Runs first 21 subjects in the participants.tsv in parallel
```

For specific subjects:

```bash
sbatch --array=0-1 fmriprep.sb 101 110 # Runs these two subjects in parallel
```
