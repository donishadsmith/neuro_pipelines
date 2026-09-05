#!/bin/bash

# SLURM MANUAL https://slurm.schedmd.com/sbatch.html
# GNU https://www.gnu.org/software/bash/manual/html_node/Bash-Conditional-Expressions.html
# or do man slurm
# Run `bash workflow.sh`
# If debugging run `bash -x workflow.sh`
# Use `sed -i "s/\r//" workflow.sh` if you get a "$'\r': command not found" error

# NOTE: If a specific image template is needed but the file is not found in templateflow but the filename is there
# it is a pointer to the file and needs to be downloaded. If you have download firewall issues
# do `conda activate /projects/bigos_lab/envs/py313`
# then use `git annex whereis filename.nii.gz` get the download link, download it, and transfer via Globus
# If multiple, you can do something like `git annex whereis *.nii.gz | grep "https:" | cut -c 7-"` or
# `git annex whereis *.nii.gz | grep "https:" | cut -c 7- > links.txt`

# Pharmacological fMRI paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC5067101/

# ==============================================
# PIPELINE FLOW VARIABLES (Set to true or false)
# ==============================================
RUN_FIRST_LEVEL=true                                    # Runs the individual first level analysis
RUN_EXTRACT_INDEPENDENT_ROI_BETAS=false                 # Extracts betas from independent (a-priori) sphere ROIs
RUN_SECOND_LEVEL=true                                   # Runs the group level second level analysis
RUN_CLUSTER_RESULTS=true                                # Determines the significant clusters and outputs to a table and various masks
RUN_CLUSTER_MNI_LOCATIONS=true                          # Determines the MNI region name of each cluster and adds to table created from RUN_CLUSTER_RESULTS
RUN_EXTRACT_INDIVIDUAL_BETAS=true                       # Extracts the individual cluster betas and outputs to table; also adds interpretations

SEND_EMAILS=false                                       # Whether or not to send emails
EMAIL_ADDRESS=""                                        # Email address to report job completion

# ========================
# GLOBAL PIPELINE SETTINGS
# ========================
export FMRIPREP_VERSION="25.2.3"                        # Version of fMRIPrep being used
export ANALYSIS_TYPE="glm"                              # Choose "glm" or "gPPI"
export METHOD="parametric"                              # Choose "parametric" or "nonparametric"
export SEED_MASK_PATH=""                                # Add path if using gPPI
export INDEPENDENT_ROI_DIR=""                           # Add path of the independent ROI directory
export COHORT="kids"                                    # Choose "kids" or "adults"

# For first level
SUBJECTS_IDS=()                                         # Set to () if running all subjects and set NUM_SUBJECTS else set specific IDS (e.g. 101 102 103)
NUM_SUBJECTS=20                                         # Set to "" if using SUBJECTS_IDS else set to max number of subjects
                                                        # NOTE: NUM_SUBJECTS does not have to be exact, just as long as it is equal to or more than your
                                                        # subjects it is fine, jobs with nonexisting subjects will be cancelled automatically

# Examples TASKS=("nback" "flanker" "mtle" "mtlr" "princess")
# TASKS=("nback")
# TASKS=("nback" "flanker" "mtle" "mtlr" "princess") options for kids
# TASKS=("nback" "flanker" "mtle" "mtlr" "simplegng" "complexgng") options for adults
TASKS=("all")                                           # Set all or specific ones out of "nback", "flanker", "mtle", "mtlr", "princess"
                                                        # Can also use ("all") or "all"
# ---------------------------------------------
# FIRST LEVEL DENOISING PARAMETERS
# ---------------------------------------------
export N_MOTION_PARAMETERS=6                             # Choose 6, 12, 18, 24
export ACOMPCOR_STRATEGY="separate"                      # Choose "combined", "separate", or "none"; The "separate" options uses the white matter and CSF acompcor, so N_ACOMPCORS=5 adds 10 parameters
export N_ACOMPCORS=5                                     # Recommend choosing 5 or 6
export FD_THRESHOLD=0.5                                  # Choose a float between 0-1.0
export EXCLUSION_CRITERIA=0.3                            # Choose a float between 0.20-0.40
export FWHM=6                                            # Choose integer
export FILTER_CORRECT_TRIALS=false                       # Filter event-related tasks for correctness set to true or false

# GPPI SPECIFIC PARAMETERS
export GPPI_UPSAMPLE_DT=0.1						         # Choose float, Time resolution to upsample seed timeseries (and condition times) to prior to deconvolution. Set to "None" for no upsampling.
export GPPI_PAD_SECONDS=30.0                             # Choose float, pads timeseries for deconvolution for gPPI
export GPPI_FALTUNG_PENALTY_SYNTAX="012 0"               # Deconvolution penalty syntax. See: https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTfitter.html"

# ------------------------------------------------
# PARAMETERS FOR SECOND LEVEL
# ------------------------------------------------
export CATEGORICAL_COVARIATES="None"                    # Categorical covariates to include in model; separated by space (e.g., "sex race ethnicity"), set to "None" to use no covariates
export NUMERICAL_COVARIATES="None"                      # Numerical covariates to include in model; separated by space (e.g., "n_censored_volumes age mean_fd_after_censoring"), set to "None" to use no covariates
export GM_MASK_THRESHOLD=0.20                           # Choose a float between 0.20-0.30; Uses a gray matter probability mask to threshold group mask
export APRIORI_IMG_PATH="None"                          # Path to an apriori mask to restrict search space, set to "None" to not specify a path
export EXCLUDE_NIFTIS_FILE="None"                       # Path to a text file containing prefixes of the filename of the NIfTI images to exclude, set to "None" to not specify a path
                                                        # should contain a single column named 'nifti_prefix_filename'
export PARAMETRIC_VOXEL_CORRECTION_P=0.001              # Cluster forming threshold for parametric method
export PARAMETRIC_CLUSTER_CORRECTION_P=0.05             # Cluster correction threshold for parametric method
export PARAMETRIC_CONNECTIVITY="NN1"                    # The connectivity for parametric

export TFCE_H=2                                         # The height power for nonparametric method
export TFCE_E=0.5                                       # The extent power for nonparametric method
export NONPARAMETRIC_CONNECTIVITY=6                     # The connectivity for parametric

# ------------------------------------------------
# PARAMETERS USED WHEN GETTING THE CLUSTER RESULTS
# ------------------------------------------------
export SPHERE_RADIUS=5.0                                # Used to create masks, choose integer

# ------------------------------------------------
# PARAMETERS USED FOR IDENTIFYING CLUSTERS
# ------------------------------------------------
# Options - https://afni.nimh.nih.gov/pub/dist/doc/program_help/whereami.html
export WHEREAMI_ATLAS="FS.afni.MNI2009c_asym"

# ======================================
# *** ONLY SET PARAMETERS ABOVE THIS ***
# ======================================
[[ $SEND_EMAILS == true ]] && MAIL_ARGS=("--mail-type=END" "--mail-user=$EMAIL_ADDRESS") || MAIL_ARGS=()

if [[ $COHORT == "kids" ]]; then
    export TEMPLATE_SPACE="MNIPediatricAsym_cohort-1_res-2"

    TEMPLATE_FLOW_PATH="/projects/bigos_lab/templateflow/tpl-MNIPediatricAsym/cohort-1"
    export GM_PROBSEG_IMG_PATH="$TEMPLATE_FLOW_PATH/tpl-${TEMPLATE_SPACE}_label-GM_probseg.nii.gz"
    export TEMPLATE_MASK_PATH="$TEMPLATE_FLOW_PATH/tpl-${TEMPLATE_SPACE}_desc-brain_mask.nii.gz"
    export TEMPLATE_IMG_PATH="$TEMPLATE_FLOW_PATH/tpl-${TEMPLATE_SPACE%2}1_desc-brain_T1w.nii.gz"
    if [[ $TASKS == "all" ]]; then
        TASKS=("nback" "flanker" "mtle" "mtlr" "princess")
    fi
else
    export TEMPLATE_SPACE="MNI152NLin2009cAsym_res-2"

    BASE_SPACE="MNI152NLin2009cAsym"
    TEMPLATE_FLOW_PATH="/projects/bigos_lab/templateflow/tpl-MNI152NLin2009cAsym"
    export GM_PROBSEG_IMG_PATH="$TEMPLATE_FLOW_PATH/tpl-${BASE_SPACE}_res-02_label-GM_probseg.nii.gz"
    export TEMPLATE_MASK_PATH="$TEMPLATE_FLOW_PATH/tpl-${BASE_SPACE}_res-02_desc-brain_mask.nii.gz"
    export TEMPLATE_IMG_PATH="$TEMPLATE_FLOW_PATH/tpl-${BASE_SPACE}_res-01_desc-brain_T1w.nii.gz"
    if [[ $TASKS == "all" ]]; then
        TASKS=("nback" "flanker" "mtle" "mtlr" "simplegng" "complexgng")
    fi
fi

if [[ $ANALYSIS_TYPE == "gPPI" ]]; then
    if [[ -z $SEED_MASK_PATH ]]; then
        echo "SEED_MASK_PATH must be set when ANALYSIS_TYPE='gPPI'"
        exit 1
    fi

    if [[ ! -f $SEED_MASK_PATH ]]; then
        echo "The following SEED_MASK_PATH does not exist: $SEED_MASK_PATH"
        exit 1
    fi

    if [[ $SEED_MASK_PATH != *"sphere_mask"* ]]; then
	    echo "'sphere_mask' not in $SEED_MASK_PATH. Ignore message if intentional."
    fi

    FIRST_LEVEL_SCRIPT="first_level_gPPI.sb"
else
    FIRST_LEVEL_SCRIPT="first_level_glm.sb"
fi

if [[ $RUN_EXTRACT_INDEPENDENT_ROI_BETAS == true ]]; then
    if [[ -z $INDEPENDENT_ROI_DIR ]]; then
        echo "INDEPENDENT_ROI_DIR must be set when RUN_EXTRACT_INDEPENDENT_ROI_BETAS is true"
        exit 1
    fi
fi

[[ ${#SUBJECTS_IDS[@]} -eq 0 ]] && N_SUBJECTS=$(( $NUM_SUBJECTS -1 )) || N_SUBJECTS=$(( ${#SUBJECTS_IDS[@]} -1 ))

for CURRENT_TASK in "${TASKS[@]}"; do
    export TASK=$CURRENT_TASK

    if [[ $CURRENT_TASK == "nback"  && $COHORT == "kids" ]]; then
        if [[ $ANALYSIS_TYPE == "glm" ]]; then
            FIRST_LEVEL_GLT_LABELS=("1-back_vs_center" "2-back_vs_center" "2-back_vs_1-back")
        else
            FIRST_LEVEL_GLT_LABELS=("PPI_1-back_vs_PPI_center" "PPI_2-back_vs_PPI_center" "PPI_2-back_vs_PPI_1-back")
        fi
    elif [[ $CURRENT_TASK == "flanker" ]]; then
        if [[ $ANALYSIS_TYPE == "glm" ]]; then
            FIRST_LEVEL_GLT_LABELS=("incongruent_vs_congruent" "nogo_vs_neutral" "nogo" "incongruent" "congruent")
        else
            FIRST_LEVEL_GLT_LABELS=("PPI_incongruent_vs_PPI_congruent" "PPI_nogo_vs_PPI_neutral" "PPI_nogo" "PPI_incongruent" "PPI_congruent")
        fi
    elif [[ ($CURRENT_TASK == "mtle" || $CURRENT_TASK == "mtlr") && $COHORT == "adults" ]]; then
        if [[ $CURRENT_TASK == "mtle" ]]; then
            [[ $ANALYSIS_TYPE == "glm" ]] && FIRST_LEVEL_GLT_LABELS=("aversive_encoding_vs_neutral_encoding" "neutral_encoding") || FIRST_LEVEL_GLT_LABELS=("PPI_aversive_encoding_vs_PPI_neutral_encoding" "PPI_neutral_encoding")

        else
            [[ $ANALYSIS_TYPE == "glm" ]] && FIRST_LEVEL_GLT_LABELS=("aversive_retrieval_vs_neutral_retrieval" "neutral_retrieval") || FIRST_LEVEL_GLT_LABELS=("PPI_aversive_retrieval_vs_PPI_neutral_retrieval" "PPI_neutral_retrieval")
        fi
    else
        FIRST_LEVEL_GLT_LABELS=("placeholder")
    fi

    JOB_ID_1=""
    JOB_ID_ROI=""
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
    # TROUBLESHOOTING NODE FAILURES:
    # First-level jobs run as parallel SLURM arrays and are most likely to encounter
    # node-level issues, resulting in incomplete processing/failures (e.g.,
    # account issues with Apptainer). SLURM appears to still mark these as COMPLETED
    # even if Apptainer failed on the node.
    #
    # After first-level jobs finish, check for failures:
    # Use the following command:
    #  cd /projects/bigos_lab/mph-study/code/logs && grep -rl "Exit Code: 1\|FATAL" --include="*first_level*.err" . | sed 's/\.err$/.out/' | xargs -I{} head {} | grep "Hostname:\|Subject:"
    #
    # If there are errors it will output text of the host (e.g., cpu104) with issue and the subject
    # Then in first_level_glm.sb or first_level_gPPI.sb add:
    #  #SBATCH --exclude=cpu104,cpu108
    #
    # to exclude these houses and resubmit jobs for those specific subject:
    #  sbatch --array=0-1 first_level_glm.sb 90000 90001
    if [[ $RUN_FIRST_LEVEL == true ]]; then
        JOB_ID_1=$(sbatch --parsable --array=0-$N_SUBJECTS "${MAIL_ARGS[@]}" $FIRST_LEVEL_SCRIPT "${SUBJECTS_IDS[@]}")

        echo -e "- FIRST LEVEL JOB SUBMITTED (JOB ID: $JOB_ID_1)\n"
    else
        echo -e "- SKIPPING FIRST LEVEL JOB\n"
    fi

    # =================================
    # RUN_EXTRACT_INDEPENDENT_ROI_BETAS
    # =================================
    if [[ $RUN_EXTRACT_INDEPENDENT_ROI_BETAS == true ]]; then
        [[ -n $JOB_ID_1 ]] && DEPENDENCY_STR="--dependency=afterany:$JOB_ID_1" || DEPENDENCY_STR=""
        JOB_ID_ROI=$(sbatch --parsable $DEPENDENCY_STR --array=0 "${MAIL_ARGS[@]}" extract_independent_roi_betas.sb $CURRENT_TASK)

        echo -e "- EXTRACT INDEPENDENT ROI BETAS SUBMITTED (JOB ID: $JOB_ID_ROI)\n"
    else
        echo -e "- SKIPPING EXTRACT INDEPENDENT ROI BETAS JOB\n"
    fi

    # =======================================================
    # RUN_SECOND_LEVEL (WITH PARALLEL FIRST_LEVEL_GLT_LABELS)
    # =======================================================
    if [[ $RUN_SECOND_LEVEL == true ]]; then
        for LABEL in ${FIRST_LEVEL_GLT_LABELS[@]}; do
            [[ $LABEL != "placeholder" ]] && { export FIRST_LEVEL_GLT_LABEL=$LABEL; TEXT_STR="FOR $LABEL"; } || { export FIRST_LEVEL_GLT_LABEL=""; TEXT_STR=""; }

            [[ -n $JOB_ID_1 ]] && DEPENDENCY_STR="--dependency=afterany:$JOB_ID_1" || DEPENDENCY_STR=""
            JOB_ID_2=$(sbatch --parsable $DEPENDENCY_STR --array=0 "${MAIL_ARGS[@]}" second_level.sb $CURRENT_TASK)

            echo -e "- SECOND LEVEL SUBMITTED $TEXT_STR (JOB ID: $JOB_ID_2)\n" | tr -s " "

            SECOND_LEVEL_JOB_IDS=${SECOND_LEVEL_JOB_IDS:+${SECOND_LEVEL_JOB_IDS}:}${JOB_ID_2}
        done
    else
        echo -e "- SKIPPING SECOND LEVEL JOB\n"
    fi

    # ===================
    # RUN_CLUSTER_RESULTS
    # ===================
    if [[ $RUN_CLUSTER_RESULTS == true ]]; then
        [[ -n $SECOND_LEVEL_JOB_IDS ]] && DEPENDENCY_STR="--dependency=afterany:$SECOND_LEVEL_JOB_IDS" || DEPENDENCY_STR=""
        JOB_ID_3=$(sbatch --parsable $DEPENDENCY_STR --array=0 "${MAIL_ARGS[@]}" get_cluster_results.sb $CURRENT_TASK)

        echo -e "- GET CLUSTER RESULTS SUBMITTED (JOB ID: $JOB_ID_3)\n"
    else
        echo -e "- SKIPPING GET CLUSTER RESULTS JOB\n"
    fi

    # =========================
    # RUN_CLUSTER_MNI_LOCATIONS
    # =========================
    if [[ $RUN_CLUSTER_MNI_LOCATIONS == true ]]; then
        [[ -n $JOB_ID_3 ]] && DEPENDENCY_STR="--dependency=afterany:$JOB_ID_3" || DEPENDENCY_STR=""
        JOB_ID_4=$(sbatch --parsable $DEPENDENCY_STR --array=0 "${MAIL_ARGS[@]}" identify_cluster_locations.sb $CURRENT_TASK)

        echo -e "- IDENTIFY MNI LOCATIONS SUBMITTED (JOB ID: $JOB_ID_4)\n"
    else
        echo -e "- SKIPPING IDENTIFY MNI LOCATIONS JOB\n"
    fi

    # ============================
    # RUN_EXTRACT_INDIVIDUAL_BETAS
    # ============================
    if [[ $RUN_EXTRACT_INDIVIDUAL_BETAS == true ]]; then
        [[ -n $JOB_ID_4 ]] && DEPENDENCY_STR="--dependency=afterany:$JOB_ID_4" || DEPENDENCY_STR=""
        JOB_ID_5=$(sbatch --parsable $DEPENDENCY_STR --array=0 "${MAIL_ARGS[@]}" extract_individual_betas.sb $CURRENT_TASK)

        echo -e "- EXTRACT INDIVIDUAL BETAS SUBMITTED (JOB ID: $JOB_ID_5)\n"
    else
        echo -e "- SKIPPING EXTRACT INDIVIDUAL BETAS JOB\n"
    fi
done
