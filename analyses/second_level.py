import argparse, shutil, subprocess, sys
from functools import lru_cache
from pathlib import Path

import bids, nibabel as nib, numpy as np, pandas as pd
from nilearn.masking import intersect_masks
from nilearn.image import resample_to_img

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger
from nifti2bids.qc import get_n_censored_volumes

from _denoising import remove_collinear_columns
from _utils import (
    drop_dose_rows,
    get_contrast_entity_key,
    get_first_level_gltsym_codes,
    get_second_level_glt_codes,
    get_nontarget_dose,
    get_interpretation_labels,
    estimate_noise_smoothness,
    perform_cluster_simulation,
    threshold_palm_output,
)

LGR = setup_logger(__name__)

EXCLUDE_COLS = ["participant_id", "session_id", "InputFile", "dose"]
CATEGORICAL_VARS = set(["sex", "race", "ethnicity"])
SUBJECT_CONSTANT_VARS = ["age"] + list(CATEGORICAL_VARS)
# From most to least
DEPRIORITIZED_REGRESSORS = ["ethnicity", "race", "sex", "age", "n_censored_volumes"]

GLT_CODES = (
    "-gltCode 5_vs_0 'dose : 1*'5' -1*'0'' ",
    "-gltCode 10_vs_0 'dose : 1*'10' -1*'0'' ",
    "-gltCode 10_vs_5 'dose : 1*'10' -1*'5'' ",
    "-gltCode 0 'dose : 1*'0'' ",
    "-gltCode 5 'dose : 1*'5'' ",
    "-gltCode 10 'dose : 1*'10'' ",
    "-gltCode mean 'dose : {mean_code}' ",
)


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Perform second level analysis. Cool paper: https://onlinelibrary.wiley.com/doi/10.1002/hbm.70437"
    )
    parser.add_argument(
        "--bids_dir", dest="bids_dir", required=True, help="Path to BIDS directory."
    )
    parser.add_argument(
        "--deriv_dir",
        dest="deriv_dir",
        required=False,
        default=None,
        help="Root of the derivatives directory.",
    )
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help=(
            "Path to directory containing the extracted beta coefficients. "
            "Files are grabbed recursively, only the naming of the coefficients "
            "need to be BIDS compliant."
        ),
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination directory for analysis.",
    )
    parser.add_argument(
        "--space",
        dest="space",
        default="MNIPediatricAsym_cohort-1_res-2",
        required=False,
        help="Template space.",
    )
    parser.add_argument(
        "--gm_probseg_img_path",
        dest="gm_probseg_img_path",
        default=None,
        required=False,
        help=(
            "The probability mask for gray matter voxels. Should be in same space as the template."
            "Used to exclude non-gray matter voxels in the group mask and exclude false activations from "
            "white matter and ventricles. http://jpeelle.net/mri/misc/creating_explicit_mask.html"
        ),
    )
    parser.add_argument(
        "--gm_mask_threshold",
        dest="gm_mask_threshold",
        default=0.20,
        type=float,
        required=False,
        help="The probability for gray matter voxels in the mask. See: https://pmc.ncbi.nlm.nih.gov/articles/PMC3812339/",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--first_level_glt_label",
        dest="first_level_glt_label",
        required=False,
        default=None,
        help=(
            "Name of a single valid first level gltlabel for the given task. "
            "Used to run several labels of a task in parallel if there are many. "
            "Particularly useful is using nonparametric."
        ),
    )
    parser.add_argument(
        "--analysis_type",
        dest="analysis_type",
        choices=["glm", "gPPI"],
        required=True,
        help="The type of analysis performed (glm or gPPI).",
    )
    parser.add_argument(
        "--group_mask_threshold",
        dest="group_mask_threshold",
        default=1.0,
        required=False,
        type=float,
        help="Value between 0 to 1 denoting the level of intersection for the masks.",
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=False,
        default=None,
        help=("Path to Apptainer image of Afni with R. Required if using parametric."),
    )
    parser.add_argument(
        "--fsl_img_path",
        dest="fsl_img_path",
        required=False,
        default=None,
        help=(
            "Path to apptainer image of FSL with Palm using Octave. "
            "Required if method is nonparametric, palm is not in path, "
            "matlab not module and fsl not module"
        ),
    )
    parser.add_argument(
        "--exclude_covariates",
        dest="exclude_covariates",
        default=None,
        required=False,
        nargs="+",
        help=(
            "Additional covariates to exclude from second level model. Each covariate should be in a single string "
            "separated by space"
        ),
    )
    parser.add_argument(
        "--method",
        dest="method",
        default="nonparametric",
        choices=["parametric", "nonparametric"],
        required=False,
        help=(
            "Whether to use 3dlmer (parametric) or Palm (nonparametric). "
            "Typically better to use nonparametric, it doesn't assume the error distribution of the "
            "data and better controls false positives. Refer to: https://www.pnas.org/doi/abs/10.1073/pnas.1602413113?utm_source "
            "and https://onlinelibrary.wiley.com/doi/10.1002/hbm.23115. If parametric is used then the acf method method should "
            "Nonparametric is stochastic be used on the residuals to estimate smoothness and determine the appropriate cluster "
            "size via simulations. however Palm sets seed to 0 by default for reproducibility. Though different results "
            "are expected if using Octave: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)UserGuide.html"
        ),
    )
    parser.add_argument(
        "--n_permutations",
        dest="n_permutations",
        default="auto",
        required=False,
        help=(
            "If method is nonparametric, the number of permutations to pass to Palm. "
            "For 'auto' the permutation is computed by doing 2^N since each contrast be "
            "either positive or negative in the two sample paired case or the single sample test. "
            "If the max permutation exceeds 10,000, then the permutation is set to 10,000. "
            "Lowest p-value -log10(1/1e4) in that case."
        ),
    )
    parser.add_argument(
        "--tfce_H",
        dest="tfce_H",
        default=2,
        required=False,
        help=(
            "The height power. Higher values weigh signal intensity more. "
            "See https://www.fmrib.ox.ac.uk/datasets/techrep/tr08ss1/tr08ss1.pdf"
        ),
    )
    parser.add_argument(
        "--tfce_E",
        dest="tfce_E",
        default=0.5,
        required=False,
        help=(
            "The extent power. Higher values weigh cluster extent more. "
            "See https://www.fmrib.ox.ac.uk/datasets/techrep/tr08ss1/tr08ss1.pdf"
        ),
    )
    parser.add_argument(
        "--tfce_C",
        dest="tfce_C",
        default=6,
        required=False,
        help=(
            "The connectivity to use for the nonparametric approach. "
            "See https://www.fmrib.ox.ac.uk/datasets/techrep/tr08ss1/tr08ss1.pdf"
        ),
    )
    parser.add_argument(
        "--cluster_correction_p",
        dest="cluster_correction_p",
        default=0.05,
        type=float,
        required=False,
        help=(
            "Significance threshold for cluster significance for nonparametric. "
            "This script uses the threshold free cluster enhancement approach which "
            "eliminates the need to select an arbritrary threshold for the voxels (cluster-forming threshold)"
        ),
    )
    parser.add_argument(
        "--n_cores",
        dest="n_cores",
        default=1,
        type=int,
        required=False,
        help="Number of cores to use for 3dlmer when method is parametric.",
    )
    parser.add_argument(
        "--exclude_niftis_file",
        dest="exclude_niftis_file",
        default=None,
        required=False,
        help=(
            "Prefixes of the filename of the NIfTI images to exclude. "
            "Should contain a single column named 'nifti_prefix_filename'."
        ),
    )

    return parser


def append_global_exclude_covariates(exclude_covariates):
    if exclude_covariates:
        global EXCLUDE_COLS
        EXCLUDE_COLS.extend(exclude_covariates)

        LGR.info(
            f"Added the following to the EXCLUDE_COLS global variable: {exclude_covariates}"
        )


def get_beta_files(analysis_dir, task, first_level_glt_label):
    return sorted(
        list(
            Path(analysis_dir).rglob(
                f"*{task}*_desc-{first_level_glt_label}_betas.nii.gz"
            )
        )
    )


def exclude_beta_files(beta_files, exclude_niftis_file):
    if not exclude_niftis_file:
        return beta_files

    df = pd.read_csv(exclude_niftis_file, sep=None, engine="python")
    excluded_niftis_prefixes = [
        Path(nifti_prefix_filename).name.split("_desc")[0]
        for nifti_prefix_filename in df["nifti_prefix_filename"].tolist()
    ]

    return [
        beta_file
        for beta_file in beta_files
        if Path(beta_file).name.split("_space")[0] not in excluded_niftis_prefixes
    ]


def get_subjects(beta_files):
    return sorted([get_entity_value(file.name, "sub") for file in beta_files])


def drop_constant_columns(data_table):
    constant_columns = [
        col for col in data_table.columns if data_table[col].nunique() == 1
    ]

    LGR.info(f"Dropping the following constant columns: {constant_columns}")

    return data_table.drop(columns=constant_columns)


def create_data_table(bids_dir, subject_list, beta_files):
    bids_dir = Path(bids_dir)
    participants_df = pd.read_csv(bids_dir / "participants.tsv", sep="\t")

    session_files = sorted(list(bids_dir.rglob("sub-*_sessions.tsv")))
    sessions_dfs = []
    for session_file in session_files:
        if (sub_id := get_entity_value(session_file, "sub")) not in subject_list:
            continue

        df = pd.read_csv(session_file, sep="\t")
        df["participant_id"] = f"sub-{sub_id}"

        subject_beta_files = [str(file) for file in beta_files if sub_id in str(file)]
        for subject_beta_file in subject_beta_files:
            ses_id = get_entity_value(
                subject_beta_file, "ses", return_entity_prefix=True
            )
            df.loc[df["session_id"] == ses_id, "InputFile"] = subject_beta_file
            censor_file = (
                Path(subject_beta_file).name.split("desc-")[0] + "desc-censor.1D"
            )
            parent_path = Path(subject_beta_file).parent
            if parent_path.name == "betas":
                parent_path = parent_path.parent

            censor_file = parent_path / censor_file
            if censor_file.exists():
                df.loc[df["session_id"] == ses_id, "n_censored_volumes"] = (
                    get_n_censored_volumes(censor_file)
                )
            else:
                df.loc[df["session_id"] == ses_id, "n_censored_volumes"] = np.nan

        sessions_dfs.append(df)

    all_sessions = pd.concat(sessions_dfs, ignore_index=True)
    data_table = all_sessions.merge(participants_df, on="participant_id")

    for col in ["acq_time", "acq_date"]:
        if col in data_table.columns:
            data_table = data_table.drop(col, axis=1)

    column_names = (
        ["participant_id", "dose"]
        + [
            name
            for name in data_table.columns
            if name not in ["participant_id", "dose", "InputFile"]
        ]
        + ["InputFile"]
    )
    data_table = data_table.loc[:, column_names]
    data_table = data_table.dropna(how="all", axis=1).dropna(axis=0)
    data_table["dose"] = data_table["dose"].astype(int).astype(str)
    # Make life easier conceptually by organizing data by dose: smallest -> largest
    data_table = data_table.sort_values(by="dose", ascending=True)

    exclude = list(CATEGORICAL_VARS) + EXCLUDE_COLS
    continuous_vars = set(data_table.columns).difference(exclude)
    for continuous_var in continuous_vars:
        data_table[continuous_var] = data_table[continuous_var].astype(float)

    return drop_constant_columns(data_table)


@lru_cache()
def get_layout(bids_dir, deriv_dir):
    return bids.BIDSLayout(bids_dir, derivatives=deriv_dir or None)


def create_group_mask(
    dst_dir,
    layout,
    task,
    space,
    group_mask_threshold,
    filtered_beta_files,
    gm_probseg_img_path,
    gm_mask_threshold,
    method,
    entity_key,
    first_level_glt_label,
    second_level_glt_code=None,
):
    if second_level_glt_code:
        group_mask_filename = (
            dst_dir
            / "group_masks"
            / method
            / f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{second_level_glt_code}_desc-group_mask.nii.gz"
        )
    else:
        group_mask_filename = (
            dst_dir
            / "group_masks"
            / method
            / f"task-{task}_{entity_key}-{first_level_glt_label}_desc-group_mask.nii.gz"
        )
    group_mask_filename.parent.mkdir(parents=True, exist_ok=True)

    subject_mask_files = []
    for filtered_beta_file in filtered_beta_files:
        sub_id = get_entity_value(filtered_beta_file, "sub")
        ses_id = get_entity_value(filtered_beta_file, "ses")

        mask_files = layout.get(
            scope="derivatives",
            subject=sub_id,
            session=ses_id,
            task=task,
            suffix="mask",
            extension="nii.gz",
            return_type="file",
        )

        mask_files = [mask_file for mask_file in mask_files if space in str(mask_file)]
        subject_mask_files.extend(mask_files)

    group_mask = intersect_masks(subject_mask_files, threshold=group_mask_threshold)
    group_mask = create_gm_only_group_mask(
        gm_probseg_img_path, gm_mask_threshold, group_mask
    )

    nib.save(group_mask, group_mask_filename)

    return group_mask_filename


def create_gm_only_group_mask(gm_probseg_img_path, gm_mask_threshold, group_mask):
    if not gm_probseg_img_path:
        return group_mask

    LGR.info(
        "Thresholding GM probability mask, any voxel with a proability greater than "
        f"{gm_mask_threshold} will be retained in the group mask."
    )

    gm_prob_img = nib.load(gm_probseg_img_path)
    resmapled_gm_prob_img = resample_to_img(
        gm_prob_img, group_mask, interpolation="nearest"
    )
    binarized_gm_image = resmapled_gm_prob_img.get_fdata() > gm_mask_threshold

    group_mask_data = group_mask.get_fdata() * binarized_gm_image

    return nib.nifti1.Nifti1Image(group_mask_data, group_mask.affine, group_mask.header)


def get_glt_codes_str(data_table):
    glt_str = ""
    available_doses = sorted(data_table["dose"].astype(str).unique())
    for glt_code in GLT_CODES:
        level_str = glt_code.removeprefix("-gltCode").lstrip().split(" ")[0]
        if level_str == "mean":
            value = round(1 / len(available_doses), 4)
            dose_list = [f"'{x}'" for x in available_doses]
            mean_code = f"{value}*" + f" +{value}*".join(dose_list)
            glt_str += glt_code.format(mean_code=mean_code)
        elif "_vs_" not in glt_code:
            glt_str += glt_code if level_str in available_doses else ""
        else:
            dose_list = level_str.split("_vs_")
            if all(dose in available_doses for dose in dose_list):
                glt_str += glt_code

    return glt_str


def get_model_str(data_table):
    exclude = set(EXCLUDE_COLS).difference(["dose"])
    columns = [col for col in data_table.columns if col not in exclude]

    model_str = "+".join(columns)
    model_str += "+(1|participant_id)"

    LGR.info(f"The following model will be used: {model_str}")

    return model_str


def get_centering_str(data_table):
    exclude = list(CATEGORICAL_VARS) + EXCLUDE_COLS
    continuous_vars = set(data_table.columns).difference(exclude)
    if not continuous_vars:
        return ""

    qvars_str = "'" + ",".join(continuous_vars) + "'"
    centers_str = "'" + ",".join(["0"] * len(continuous_vars)) + "'"
    centering_str = f"-qVars {qvars_str} -qVarCenters {centers_str}"

    LGR.info(f"The following centering string will be used: {centering_str}")

    return centering_str


def generate_matrices_filenames(
    output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
):
    matrices_filenames_dict = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = (
        f"task-{task}_{entity_key}-{first_level_glt_label}"
        f"_gltcode-{second_level_glt_code}"
    )

    if "_vs_" in second_level_glt_code:
        matrices_filenames_dict["eb_file"] = (
            output_dir / f"{prefix}_desc-exchangeability_blocks.csv"
        )
    else:
        matrices_filenames_dict["eb_file"] = None

    matrices_filenames_dict["design_matrix_file"] = (
        output_dir / f"{prefix}_desc-design_matrix.csv"
    )
    matrices_filenames_dict["header_file"] = (
        output_dir / f"{prefix}_desc-header_names.txt"
    )
    matrices_filenames_dict["contrast_matrix_file"] = (
        output_dir / f"{prefix}_desc-contrast_matrix.csv"
    )

    return matrices_filenames_dict


def prioritize_regressors(design_matrix):
    if design_matrix.shape[0] <= design_matrix.shape[1]:
        for regressor in DEPRIORITIZED_REGRESSORS:
            if drop_columns := [
                col for col in design_matrix.columns if col.startswith(regressor)
            ]:
                LGR.info(
                    f"Dropping the following regressor(s) to save dof: {drop_columns}"
                )
                design_matrix = design_matrix.drop(columns=drop_columns)

            LGR.info(
                f"N OBSERVATIONS: {design_matrix.shape[0]}; N COLUMNS: {design_matrix.shape[0]}"
            )
            if design_matrix.shape[0] >= design_matrix.shape[1]:
                break

    return design_matrix


def create_design_matrix(
    glt_data_table,
    include_intercept,
    average_within_subjects=False,
    second_level_glt_code=None,
):
    categorical_cols = [
        col for col in glt_data_table.columns if col in CATEGORICAL_VARS
    ]

    continuous_cols = [
        col
        for col in glt_data_table.columns
        if col not in EXCLUDE_COLS and col not in categorical_cols
    ]

    for col in categorical_cols:
        glt_data_table[col] = glt_data_table[col].astype(str)

    # Assumed to be the mean glt code
    if average_within_subjects:
        numeric_table = (
            glt_data_table.groupby("participant_id")
            .mean(numeric_only=True)
            .reset_index(drop=True)
        )
        if "participant_id" in numeric_table.columns:
            numeric_table = numeric_table.drop(columns=["participant_id"])

        categorical_table = (
            glt_data_table.groupby("participant_id")[categorical_cols]
            .first()
            .reset_index(drop=True)
        )
        if "participant_id" in categorical_table.columns:
            categorical_table = categorical_table.drop(columns=["participant_id"])

        glt_data_table = pd.concat([numeric_table, categorical_table], axis=1)

    design_components = {}
    mean_center = lambda arr: arr - arr.mean()
    for col in continuous_cols:
        design_components.update({col: mean_center(glt_data_table[col].to_numpy())})

    # All categorical variables are constant across sessions and have been dropped
    # previously for dose pairisons to avoid rank deficiency
    # Categorical cols need to be mean centered so that no group can be coded as 0
    # If one categorical variable is binary (0 and 1), then
    # for 0 E[Y] = B0 + B1E[Xc] = E[Y] = B0 + B1(0), which forces the intercept to be
    # the reference group
    for col in categorical_cols:
        dummies = pd.get_dummies(
            glt_data_table[col], prefix=col, drop_first=True
        ).astype(int)
        design_components.update(
            {col: mean_center(dummies[col].to_numpy()) for col in dummies.columns}
        )

    design_matrix = pd.DataFrame(design_components)
    if design_matrix.shape[1] > 1:
        regressor_positions = {
            index: col for index, col in enumerate(design_matrix.columns)
        }
        design_arr, regressor_positions = remove_collinear_columns(
            design_matrix.to_numpy(), regressor_positions
        )
        design_matrix = pd.DataFrame(
            design_arr, columns=list(regressor_positions.values())
        )
        design_matrix = prioritize_regressors(design_matrix)

    if include_intercept:
        intercept = [1] * design_matrix.shape[0]
        design_matrix.insert(0, "intercept", intercept)
    else:
        first_label, _ = get_interpretation_labels(second_level_glt_code)
        dose_codes = np.where(glt_data_table["dose"].astype(str) == first_label, 1, -1)
        design_matrix.insert(0, "dose", dose_codes)
        subject_regressors = pd.get_dummies(
            glt_data_table["participant_id"], prefix="", prefix_sep=""
        ).astype(int)
        design_matrix = pd.concat(
            [
                design_matrix.reset_index(drop=True),
                subject_regressors.reset_index(drop=True),
            ],
            axis=1,
        )

    return design_matrix


def create_contrast_matrix(design_matrix, contrast_matrix_filename):
    vector_pos = np.zeros(design_matrix.shape[1])
    vector_pos[0] = 1
    # Zeroes will show up as negative, that is fine -0 == 0 returns True
    vector_neg = vector_pos * -1

    np.savetxt(
        contrast_matrix_filename,
        np.array([vector_pos, vector_neg]),
        delimiter=",",
        fmt="%.4f",
    )


def create_header_file(design_matrix, header_file):
    with open(header_file, "w") as f:
        f.write(f"{','.join(design_matrix.columns.tolist())}")


def write_contrast_direction_file(
    output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
):
    prefix = f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{second_level_glt_code}"
    contrast_direction_file = output_dir / f"{prefix}_desc-contrast_direction.txt"

    if "_vs_" in second_level_glt_code:
        first_label, second_label = get_interpretation_labels(second_level_glt_code)
        positive_label = f"{first_label} > {second_label}"
        negative_label = f"{second_label} > {first_label}"
    else:
        positive_label = "above mean of 0"
        negative_label = "below mean of 0"

    with open(contrast_direction_file, "w") as f:
        f.write("# Positive direction contrast:\n")
        f.write(f"c1: {positive_label}\n")
        f.write("\n# Negative direction contrast:\n")
        f.write(f"c2: {negative_label}\n")


def create_mean_matrices(
    glt_data_table,
    output_dir,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
):
    """
    Takes the data table and creates matrices for PALM specifically for means
    (e.g., 5, mean, etc).

    # PALM GUIDE: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)UserGuide.html

    According to the PALM guide, FSL design matrices need to be used.
    This is the guide for FSL design matrices:
    https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/glm.html?h=design
    """
    matrices_filenames_dict = generate_matrices_filenames(
        output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
    )

    design_matrix = create_design_matrix(
        glt_data_table,
        include_intercept=True,
        average_within_subjects=(second_level_glt_code == "mean"),
    )
    design_matrix.to_csv(
        matrices_filenames_dict["design_matrix_file"],
        sep=",",
        header=False,
        index=False,
    )
    create_header_file(design_matrix, matrices_filenames_dict["header_file"])

    create_contrast_matrix(
        design_matrix, matrices_filenames_dict["contrast_matrix_file"]
    )
    write_contrast_direction_file(
        output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
    )

    return matrices_filenames_dict


def create_comparison_matrices(
    glt_data_table,
    output_dir,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
):
    """
    Takes the data table and creates matrices for PALM specifically for comparisons
    (e.g., 5_vs_0).

    # PALM GUIDE: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)UserGuide.html

    According to the PALM guide, FSL design matrices need to be used.
    This is the guide for FSL design matrices:
    https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/glm.html?h=design
    """
    matrices_filenames_dict = generate_matrices_filenames(
        output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
    )

    for col in SUBJECT_CONSTANT_VARS:
        if col in glt_data_table.columns:
            glt_data_table = glt_data_table.drop(col, axis=1)

    eb_data = glt_data_table["participant_id"].factorize()[0] + 1
    np.savetxt(matrices_filenames_dict["eb_file"], eb_data, delimiter=",", fmt="%d")

    design_matrix = create_design_matrix(
        glt_data_table,
        include_intercept=False,
        second_level_glt_code=second_level_glt_code,
    )
    design_matrix.to_csv(
        matrices_filenames_dict["design_matrix_file"],
        sep=",",
        header=False,
        index=False,
    )
    create_header_file(design_matrix, matrices_filenames_dict["header_file"])

    create_contrast_matrix(
        design_matrix, matrices_filenames_dict["contrast_matrix_file"]
    )
    write_contrast_direction_file(
        output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
    )

    return matrices_filenames_dict


def compute_n_permutation(glt_data_table):
    n_subjects = len(glt_data_table["participant_id"].unique())
    max_permutation = 2**n_subjects
    LGR.info(f"Maximum permutations possible: {max_permutation}")

    n_permutations = min(max_permutation, 10000)
    LGR.info(f"Setting number of permutations to: {n_permutations}")

    return n_permutations


def create_concatenated_image(
    output_dir,
    glt_data_table,
    fsl_img_path,
    use_native_fsl,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{second_level_glt_code}"
    concatenated_filename = output_dir / f"{prefix}_desc-group_concatenated.nii.gz"
    if concatenated_filename.exists():
        concatenated_filename.unlink()

    fsl_merge_call = (
        "fslmerge"
        if use_native_fsl
        else f"apptainer exec -B /projects:/projects {fsl_img_path} fslmerge"
    )

    fsl_maths_call = (
        "fslmaths"
        if use_native_fsl
        else f"apptainer exec -B /projects:/projects {fsl_img_path} fslmaths"
    )

    if second_level_glt_code == "mean":
        LGR.info("Averaging doses within subjects for the mean contrast.")

        subject_mean_images = []
        for subject, group in glt_data_table.groupby("participant_id"):
            subject_files = group["InputFile"].tolist()
            subject_temp_merged = (
                concatenated_filename.parent / f"temp_{subject}_merged.nii.gz"
            )
            subject_mean_img = (
                concatenated_filename.parent / f"temp_{subject}_mean.nii.gz"
            )

            subprocess.run(
                f"{fsl_merge_call} -t {subject_temp_merged} {' '.join(subject_files)}",
                shell=True,
                check=True,
            )
            subprocess.run(
                f"{fsl_maths_call} {subject_temp_merged} -Tmean {subject_mean_img}",
                shell=True,
                check=True,
            )

            subject_mean_images.append(str(subject_mean_img))
            subject_temp_merged.unlink()

        cmd = f"{fsl_merge_call} -t {concatenated_filename} {' '.join(subject_mean_images)}"
        subprocess.run(cmd, shell=True, check=True)

        for img in subject_mean_images:
            Path(img).unlink()

    else:
        files_to_merge = glt_data_table["InputFile"].tolist()
        cmd = f"{fsl_merge_call} -t {concatenated_filename} {' '.join(files_to_merge)}"
        LGR.info(f"Concatenating paired images: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    return concatenated_filename


def perform_palm(
    concatenated_filename,
    group_mask_filename,
    design_matrix_file,
    eb_file,
    contrast_matrix_file,
    fsl_img_path,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
    n_permutations,
    tfce_H,
    tfce_E,
    tfce_C,
    use_native_palm,
):
    output_prefix = (
        concatenated_filename.parent
        / f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{second_level_glt_code}_desc-nonparametric"
    )

    palm_flags = (
        "-noniiclass "
        f"-i {concatenated_filename} "
        f"-m {group_mask_filename} "
        f"-d {design_matrix_file} "
        f"-t {contrast_matrix_file} "
        f"-n {n_permutations} "
        "-T "
        f"-tfce_H {tfce_H} "
        f"-tfce_E {tfce_E} "
        f"-tfce_C {tfce_C} "
        "-logp "
        "-savedof "
        f"-o {output_prefix}"
    )

    palm_flags += (
        f" -eb {eb_file} -ee -within" if "_vs_" in second_level_glt_code else " -ise"
    )

    if use_native_palm:
        palm_dir = Path(shutil.which("palm")).parent
        cmd = f"matlab -nodisplay -nosplash -r \"addpath('{palm_dir}'); palm {palm_flags}; exit;\""
    else:
        cmd = (
            f"apptainer exec -B /projects:/projects {fsl_img_path} "
            f"octave --eval 'palm {palm_flags}'"
        )

    LGR.info(f"Running PALM for {second_level_glt_code}: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return str(output_prefix)


def perform_3dlmer(
    task,
    entity_key,
    first_level_glt_label,
    dst_dir,
    data_table_filename,
    group_mask_filename,
    afni_img_path,
    model_str,
    center_str,
    glt_str,
    n_cores,
):
    output_filename = (
        dst_dir
        / "second_level_outputs"
        / "parametric"
        / f"task-{task}_{entity_key}-{first_level_glt_label}_desc-parametric_stats.nii.gz"
    )
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    if output_filename.exists():
        output_filename.unlink()

    residual_filename = Path(str(output_filename).replace("_stats", "_residuals"))
    if residual_filename.exists():
        residual_filename.unlink()

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dLMEr "
        f"-mask {group_mask_filename} "
        f"-model '{model_str}' "
        f"-jobs {n_cores} "
        f"{center_str} "
        "-dbgArgs "
        f"{glt_str} "
        f"-prefix {output_filename} "
        f"-resid {residual_filename} "
        f"-dataTable @{data_table_filename}"
    )

    LGR.info(f"Running 3dLMEr: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return residual_filename


def main(
    bids_dir,
    deriv_dir,
    analysis_dir,
    dst_dir,
    task,
    first_level_glt_label,
    analysis_type,
    space,
    gm_probseg_img_path,
    gm_mask_threshold,
    group_mask_threshold,
    afni_img_path,
    fsl_img_path,
    method,
    n_permutations,
    tfce_H,
    tfce_E,
    tfce_C,
    cluster_correction_p,
    n_cores,
    exclude_niftis_file,
    exclude_covariates,
):
    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir) if deriv_dir else None
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)

    LGR.info(f"TASK: {task}, METHOD: {method}")

    if first_level_glt_label:
        first_level_glt_labels = [first_level_glt_label]
    else:
        first_level_glt_labels = get_first_level_gltsym_codes(
            task, analysis_type, caller="second_level"
        )

    append_global_exclude_covariates(exclude_covariates)

    for first_level_glt_label in first_level_glt_labels:
        entity_key = get_contrast_entity_key(first_level_glt_label)
        LGR.info(f"FIRST LEVEL GLTLABEL: {first_level_glt_label}")
        beta_files = exclude_beta_files(
            get_beta_files(analysis_dir, task, first_level_glt_label),
            exclude_niftis_file,
        )

        if not beta_files:
            LGR.warning(f"No beta files found for {first_level_glt_label}")
            continue

        subject_list = get_subjects(beta_files)
        LGR.info(
            f"Found {len(beta_files)} files from {len(set(subject_list))} subjects"
        )

        data_table = create_data_table(bids_dir, subject_list, beta_files)
        data_table_filename = (
            dst_dir
            / f"task-{task}_{entity_key}-{first_level_glt_label}_desc-data_table.txt"
        )
        data_table.to_csv(data_table_filename, sep="\t", index=False, encoding="utf-8")

        if method == "parametric":
            if not afni_img_path:
                LGR.critical("`afni_img_path` is required when method is parametric.")
                sys.exit(1)

            LGR.critical(
                f"Using {len(data_table['InputFile'].tolist())} files "
                f"from {len(data_table['participant_id'].unique())} subjects for analysis "
            )
            LGR.info(f"Creating group mask with threshold: {group_mask_threshold}")
            group_mask_filename = create_group_mask(
                dst_dir,
                get_layout(bids_dir, deriv_dir),
                task,
                space,
                group_mask_threshold,
                data_table["InputFile"].tolist(),
                gm_probseg_img_path,
                gm_mask_threshold,
                method,
                entity_key,
                first_level_glt_label,
            )

            glt_str = get_glt_codes_str(data_table)
            model_str = get_model_str(data_table)
            center_str = get_centering_str(data_table)

            residual_filename = perform_3dlmer(
                task,
                entity_key,
                first_level_glt_label,
                dst_dir,
                data_table_filename,
                group_mask_filename,
                afni_img_path,
                model_str,
                center_str,
                glt_str,
                n_cores,
            )
            acf_parameters_filename = estimate_noise_smoothness(
                dst_dir,
                afni_img_path,
                group_mask_filename,
                residual_filename,
                first_level_glt_label,
            )
            perform_cluster_simulation(
                afni_img_path,
                group_mask_filename,
                acf_parameters_filename,
                first_level_glt_label,
            )
        else:
            # Nonparametric (PALM)
            use_native_palm = shutil.which("palm") is not None
            use_native_fsl = shutil.which("fslmerge") is not None

            if not fsl_img_path and not (use_native_palm or use_native_fsl):
                LGR.critical(
                    "`fsl_img_path` is required when method is nonparametric "
                    "and palm and `fslmerge` are not in path."
                )
                sys.exit(1)

            output_dir = dst_dir / "second_level_outputs" / "nonparametric"
            output_dir.mkdir(parents=True, exist_ok=True)
            for second_level_glt_code in get_second_level_glt_codes():
                LGR.info(f"Processing the following glt code: {second_level_glt_code}")

                vs_in_code = "_vs_" in second_level_glt_code
                glt_data_table = drop_dose_rows(
                    data_table, get_nontarget_dose(second_level_glt_code), vs_in_code
                )
                glt_data_table = drop_constant_columns(glt_data_table)

                if glt_data_table.empty:
                    LGR.info(
                        f"Skipping the following second level glt code: {second_level_glt_code}"
                    )
                    continue

                LGR.critical(
                    f"Using {len(data_table['InputFile'].tolist())} files "
                    f"from {len(data_table['participant_id'].unique())} subjects for analysis "
                    f"using {second_level_glt_code}"
                )
                LGR.info(f"Creating group mask with threshold: {group_mask_threshold}")
                group_mask_filename = create_group_mask(
                    dst_dir,
                    get_layout(bids_dir, deriv_dir),
                    task,
                    space,
                    group_mask_threshold,
                    data_table["InputFile"].tolist(),
                    gm_probseg_img_path,
                    gm_mask_threshold,
                    method,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                )

                matrix_creation_func = (
                    create_comparison_matrices if vs_in_code else create_mean_matrices
                )
                matrices_output_dict = matrix_creation_func(
                    glt_data_table,
                    output_dir,
                    task,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                )
                max_permutations = (
                    compute_n_permutation(glt_data_table)
                    if n_permutations == "auto"
                    else n_permutations
                )
                concatenated_filename = create_concatenated_image(
                    output_dir,
                    glt_data_table,
                    fsl_img_path,
                    use_native_fsl,
                    task,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                )
                output_prefix = perform_palm(
                    concatenated_filename,
                    group_mask_filename,
                    matrices_output_dict["design_matrix_file"],
                    matrices_output_dict["eb_file"],
                    matrices_output_dict["contrast_matrix_file"],
                    fsl_img_path,
                    task,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                    max_permutations,
                    tfce_H,
                    tfce_E,
                    tfce_C,
                    use_native_palm,
                )
                threshold_palm_output(
                    output_prefix,
                    second_level_glt_code,
                    cluster_correction_p,
                )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
