# First Level and Second Level Analysis

**Notes:**
- Requires an Apptainer/Singularity image of AFNI with R.
- On Windows, you may need to convert line endings from [DOS to UNIX](https://blog.bachi.net/?p=1715): ``sed -i "s/\r//" filename``

## Automated Workflow

Set your variables in ``workflow.sh``, then run:

```bash
bash workflow.sh
```

All jobs will be executed in order automatically.

## Running Scripts Independently

Edit the shell script as needed:

```bash
vim first_level_glm.sb
```

Then execute in the following order:

1. ``first_level_glm.sb`` or ``first_level_gPPI.sb``
2. ``second_level.sb``
3. ``get_cluster_results.sb``
4. ``identify_cluster_locations.sb``
5. ``extract_individual_betas.sb``

### Running All Subjects / Tasks

```bash
# First 31 subjects from participants.tsv, in parallel
sbatch --array=0-30 first_level_glm.sb
# All tasks
sbatch --array=0-4 second_level.sb
```

### Running Specific Subjects / Tasks

```bash
# Two specific subjects, in parallel
sbatch --array=0-1 first_level_glm.sb 101 102
# Two specific tasks
sbatch --array=0-1 second_level.sb nback flanker
```
