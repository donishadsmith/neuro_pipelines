# Performing first level and second level

**Notes:**
- Apptainer/Singularity image of AFNI with R required.
- May need to convert line endings from [DOS to UNIX](https://blog.bachi.net/?p=1715) for text files ``sed -i "s/\r//" filename``

```bash
sbatch --array=0-30 first_level_glm.sb # Runs first 31 subjects in the participants.tsv in parallel

sbatch --array=0-4 second_level.sb # Runs all tasks
```

For specific subjects:

```bash
sbatch --array=0-1 first_level_glm.sb 101 102 # Runs these two subjects in parallel

sbatch --array=0-1 second_level.sb nback flanker # Runs these specific tasks
```

## Order
1. first_level_glm.sb or first_level_gPPI.sb
2. second_level.sb
3. get_cluster_results.sb
4. identify_cluster_locations.sb (uses afni whereami to auto identify peak cluster coordinate location as well as how many millimeters of the closes label is to the coordinate)
5. extract_individual_betas.sb (for brain-behavior analyses and neurobiological interpretations; note the second level only tells you which clusters are different between the compared groups, so examining the first level betas from the contrast maps or condition maps are needed to determine if the difference is related to greater activation/connectivy or reduced deactivation/connectivity and in the case of the first-level contrast, will determine if the cluster is related to condition A, if the average of the betas is positive for group, or condition B if the average of the betas are negative)
