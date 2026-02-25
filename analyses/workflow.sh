
#!/bin/bash

# SLURM MANUAL https://slurm.schedmd.com/sbatch.html
# GNU https://www.gnu.org/software/bash/manual/html_node/Bash-Conditional-Expressions.html
# or do man slurm
# Run `bash workflow.sh`
# If debugging run `bash -x workflow.sh`
# Use `sed -i "s/\r//" workflow.sh` if you get a "$'\r': command not found" error

# =============================================
# CONTROL FLOW VARIABLES (Set to true or false)
# =============================================
RUN_FIRST_LEVEL=true                                    # Runs the individual first level analysis
RUN_SECOND_LEVEL=true                                   # Runs the group level second level analysis
RUN_CLUSTER_RESULTS=true                                # Determines the significant clusters and outputs to a table and various masks
RUN_CLUSTER_MNI_LOCATIONS=true                          # Determines the MNI region name of each cluster and adds to table created from RUN_CLUSTER_RESULTS
RUN_EXTRACT_INDIVIDUAL_BETAS=true                       # Extracts the individual cluster betas and outputs to table; also adds interpretations

# ========================
# GLOBAL PIPELINE SETTINGS
# ========================
export ANALYSIS_TYPE="glm"                              # Choose "glm" or "gPPI"
export METHOD="nonparametric"                           # Choose "parametric" or "nonparametric"
export SEED_MASK_PATH=""                                # Add path if using gPPI

# For first level
SUBJECTS_IDS=()                                       # Set to () if running all subjects and set NUM_SUBJECTS else set specific IDS (e.g. 101 102 103)
NUM_SUBJECTS=19                                         # Set to "" if using SUBJECTS_IDS else set to max number of subjects

# Examples TASKS=("nback" "flanker" "mtle" "mtlr" "princess")
# TASKS=("nback")
TASKS=("nback" "flanker" "mtle" "mtlr" "princess")      # Set all or specific ones out of "nback", "flanker", "mtle", "mtlr", "princess"

# --------------------------------
# FIRST LEVEL DENOISING PARAMETERS
# --------------------------------
export N_MOTION_PARAMETERS=12                           # Choose 6, 12, 18, 24
export N_ACOMPCORS=5                                    # Recommend choosing 5 or 6
export ACOMPCOR_STRATEGY="combined"                     # Choose "combined" or "separate"
export N_GLOBAL_PARAMETERS=1                            # Choose 0, 1, 2, 3, or 4
export FD_THRESHOLD=0.9                                 # Choose a float between 0-1.0
export EXCLUSION_CRITERIA=0.4                           # Choose a float between 0.20-0.40
export FWHM=6                                           # Choose integer

# ------------------------------------------------
# PARAMETERS USED WHEN GETTING THE CLUSTER RESULTS
# ------------------------------------------------
export SPHERE_RADIUS=5                                  # Used to create masks, choose integer
export TEMPLATE_MASK_PATH="/projects/bigos_lab/templateflow/tpl-MNIPediatricAsym/cohort-1/tpl-MNIPediatricAsym_cohort-1_res-2_desc-brain_mask.nii.gz" # Used to create sphere masks
export TEMPLATE_IMG_PATH="/projects/bigos_lab/templateflow/tpl-MNIPediatricAsym/cohort-1/tpl-MNIPediatricAsym_cohort-1_res-1_T1w.nii.gz" # Used for plotting

# -------------------------------------------------------
# PARAMETER USED FOR IDENTIFYING MNI LOCATION OF CLUSTERS
# -------------------------------------------------------
export AFNI_ORIENT="lpi"                                # Orientation of images -lpi/-spm -rai/-dicom
export WHEREAMI_ATLAS="Haskins_Pediatric_Nonlinear_1.0" # Options - https://afni.nimh.nih.gov/pub/dist/doc/program_help/whereami.html

# ======================================
# *** ONLY SET PARAMETERS ABOVE THIS ***
# ======================================
if [[ $ANALYSIS_TYPE == "gPPI" ]]; then

    if [[ -z $SEED_MASK_PATH ]]; then
        echo "SEED_MASK_PATH must be set when ANALYSIS_TYPE='gPPI'"
        exit 1
    fi

    if [[ ! -f $SEED_MASK_PATH ]]; then
        echo "The following SEED_MASK_PATH does not exist: $SEED_MASK_PATH"
        exit 1
    fi

    FIRST_LEVEL_SCRIPT="first_level_gPPI.sb"
else
    FIRST_LEVEL_SCRIPT="first_level_glm.sb"
fi

if [ ${#SUBJECTS_IDS[@]} -eq 0 ]; then
    N_SUBJECTS=$(( $NUM_SUBJECTS -1 ))
else
    N_SUBJECTS=$(( ${#SUBJECTS_IDS[@]} -1 ))
fi

for CURRENT_TASK in "${TASKS[@]}"; do
    export TASK=$CURRENT_TASK

    if [[ $CURRENT_TASK == "nback" ]]; then
        if [[ $ANALYSIS_TYPE == "glm" ]]; then
            FIRST_LEVEL_GLT_LABELS=("1-back_vs_0-back" "2-back_vs_0-back" "2-back_vs_1-back")
        else
            FIRST_LEVEL_GLT_LABELS=("PPI_1-back_vs_PPI_0-back" "PPI_2-back_vs_PPI_0-back" "PPI_2-back_vs_PPI_1-back")
        fi
    elif [[ $CURRENT_TASK == "flanker" ]]; then
        if [[ $ANALYSIS_TYPE == "glm" ]]; then
            FIRST_LEVEL_GLT_LABELS=("congruent_vs_neutral" "incongruent_vs_neutral" "nogo_vs_neutral" "incongruent_vs_congruent" "congruent_vs_nogo" "incongruent_vs_nogo")
        else
            FIRST_LEVEL_GLT_LABELS=("PPI_congruent_vs_PPI_neutral" "PPI_incongruent_vs_PPI_neutral" "PPI_nogo_vs_PPI_neutral" "PPI_incongruent_vs_PPI_congruent" "PPI_congruent_vs_PPI_nogo" "PPI_incongruent_vs_PPI_nogo")
        fi

    else
        FIRST_LEVEL_GLT_LABELS=("placeholder")
    fi

    JOB_ID_1=""
    SECOND_LEVEL_JOB_IDS=""
    JOB_ID_3=""
    JOB_ID_4=""
    JOB_ID_5=""

    echo "======================================================"
    echo "SUBMITTING JOBS FOR CURRENT TASK $CURRENT_TASK"
    echo -e "======================================================\n"

    # ===============
    # RUN_FIRST_LEVEL
    # ===============
    if [ $RUN_FIRST_LEVEL = true ]; then
        JOB_ID_1=$(sbatch --parsable --array=0-$N_SUBJECTS $FIRST_LEVEL_SCRIPT)
        echo -e "- FIRST LEVEL JOB SUBMITTED (JOB ID: $JOB_ID_1)\n"
    else
        echo -e "- SKIPPING FIRST LEVEL JOB\n"
    fi

    # =======================================================
    # RUN_SECOND_LEVEL (WITH PARALLEL FIRST_LEVEL_GLT_LABELS)
    # =======================================================
    if [ $RUN_SECOND_LEVEL = true ]; then
        for FIRST_LEVEL_GLT_LABEL in ${FIRST_LEVEL_GLT_LABELS[@]}; do
            if [[ $FIRST_LEVEL_GLT_LABEL != "placeholder" ]]; then
                export FIRST_LEVEL_GLTLABEL=$FIRST_LEVEL_GLT_LABEL
            else
                export FIRST_LEVEL_GLTLABEL=""
            fi

            if [[ -n $JOB_ID_1 ]]; then
                JOB_ID_2=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 --array=0 second_level.sb $CURRENT_TASK)
            else
                JOB_ID_2=$(sbatch --parsable --array=0 second_level.sb $CURRENT_TASK)
            fi

            if [[ $FIRST_LEVEL_GLT_LABEL != "placeholder" ]]; then
                echo -e "- SECOND LEVEL SUBMITTED FOR $FIRST_LEVEL_GLT_LABEL (JOB ID: $JOB_ID_2)\n"
            else
                echo -e "- SECOND LEVEL SUBMITTED (JOB ID: $JOB_ID_2)\n"
            fi

            if [[ -z $SECOND_LEVEL_JOB_IDS ]]; then
                SECOND_LEVEL_JOB_IDS=$JOB_ID_2
            else
                # APPEND the job IDs together
                SECOND_LEVEL_JOB_IDS=${SECOND_LEVEL_JOB_IDS}:${JOB_ID_2}
            fi
        done
    else
        echo -e "- SKIPPING SECOND LEVEL JOB\n"
    fi

    # ===================
    # RUN_CLUSTER_RESULTS
    # ===================
    if [ $RUN_CLUSTER_RESULTS = true ]; then
        if [[ -n $SECOND_LEVEL_JOB_IDS ]]; then
            JOB_ID_3=$(sbatch --parsable --dependency=afterok:$SECOND_LEVEL_JOB_IDS --array=0 get_cluster_results.sb $CURRENT_TASK)
        else
            JOB_ID_3=$(sbatch --parsable --array=0 get_cluster_results.sb $CURRENT_TASK)
        fi

        echo -e "- GET CLUSTER RESULTS SUBMITTED (JOB ID: $JOB_ID_3)\n"
    else
        echo -e "- SKIPPING GET CLUSTER RESULTS JOB\n"
    fi

    # =========================
    # RUN_CLUSTER_MNI_LOCATIONS
    # =========================
    if [ $RUN_CLUSTER_RESULTS = true ]; then
        if [[ -n $JOB_ID_3 ]]; then
            JOB_ID_4=$(sbatch --parsable --dependency=afterok:$JOB_ID_3 --array=0 identify_cluster_locations.sb $CURRENT_TASK)
        else
            JOB_ID_4=$(sbatch --parsable --array=0 identify_cluster_locations.sb $CURRENT_TASK)
        fi

        echo -e "- IDENTIFY MNI LOCATIONS SUBMITTED (JOB ID: $JOB_ID_4)\n"
    else
        echo -e "- SKIPPING IDENTIFY MNI LOCATIONS JOB\n"
    fi

    # ============================
    # RUN_EXTRACT_INDIVIDUAL_BETAS
    # ============================
    if [ $RUN_EXTRACT_INDIVIDUAL_BETAS = true ]; then
        if [[ -n $JOB_ID_4 ]]; then
            JOB_ID_5=$(sbatch --parsable --dependency=afterok:$JOB_ID_4 --array=0 extract_individual_betas.sb $CURRENT_TASK)
        else
            JOB_ID_5=$(sbatch --parsable --array=0 extract_individual_betas.sb $CURRENT_TASK)
        fi

        echo -e "- EXTRACT INDIVIDUAL BETAS SUBMITTED (JOB ID: $JOB_ID_5)\n"
    else
        echo -e "- EXTRACT INDIVIDUAL BETAS JOB\n"
    fi
done
