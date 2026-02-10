# Performing first level and extracting contrasts

**Notes:**
- Both "first_level.sb" and "extract_contrasts.sb" requires an Apptainer/Singularity image of AFNI with R.
- May need to convert line endings from [DOS to UNIX](https://blog.bachi.net/?p=1715) for text files ``sed -i "s/\r//" filename``

```bash
sbatch --array=0-30 first_level.sb # Runs first 31 subjects in the participants.tsv in parallel
sbatch --array=0-10 extract_contrasts.sb # Runs first 11 subjects in the participants.tsv in parallel
```

For specific subjects:

```bash
sbatch --array=0-1 first_level.sb 101 102 # Runs these two subjects in parallel
sbatch --array=0-1 extract_contrasts.sb 101 102 # Runs these two subjects in parallel
```

## Order
1. first_level.sb
2. second_level.sb
3. get_cluster_results.sb
4. identify_cluster_locations.sb (uses afni whereami to auto identify peak cluster coordinate location as well as how many millimeters of the closes label is to the coordinate)