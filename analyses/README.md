# Performing first level and second level

**Notes:**
- Apptainer/Singularity image of AFNI with R required.
- May need to convert line endings from [DOS to UNIX](https://blog.bachi.net/?p=1715) for text files ``sed -i "s/\r//" filename``

## Workflow
Set variables inside "workflow.sh", then run:

```bash
bash workflow.sh
```

All jobs will be executed in order automatically

## Run each script independently
Modify the shell script as needed and execute in the following order:

1. first_level_glm.sb or first_level_gPPI.sb
2. second_level.sb
3. get_cluster_results.sb
4. identify_cluster_locations.sb
5. extract_individual_betas.sb

```bash
sbatch --array=0-30 first_level_glm.sb # Runs first 31 subjects in the participants.tsv in parallel

sbatch --array=0-4 second_level.sb # Runs all tasks
```

For specific subjects:

```bash
sbatch --array=0-1 first_level_glm.sb 101 102 # Runs these two subjects in parallel

sbatch --array=0-1 second_level.sb nback flanker # Runs these specific tasks
```
