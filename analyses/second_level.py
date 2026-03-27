import argparse, shutil, subprocess, sys
from dataclasses import dataclass, field
from functools import lru_cache
from math import comb
from pathlib import Path

import bids, nibabel as nib, numpy as np, pandas as pd
from nilearn.masking import intersect_masks
from nilearn.image import resample_to_img

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger
from nifti2bids.io import compress_image
from nifti2bids.qc import get_n_censored_volumes

from _denoising import remove_collinear_columns
from _utils import (
    drop_dose_rows,
    get_contrast_entity_key,
    get_first_level_gltsym_codes,
    get_second_level_glt_codes,
    get_nontarget_dose,
    get_group_labels,
    is_between_group_dose_code,
    get_between_group_column,
    estimate_noise_smoothness,
    perform_cluster_simulation,
    threshold_palm_output,
)

LGR = setup_logger(__name__)


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
        "--cohort",
        dest="cohort",
        required=True,
        choices=["adults", "kids"],
        help="The cohort to analyze.",
    )
    parser.add_argument(
        "--space",
        dest="space",
        required=True,
        help="Template space (i.e. 'MNIPediatricAsym_cohort-1_res-2')",
    )
    parser.add_argument(
        "--gm_probseg_img_path",
        dest="gm_probseg_img_path",
        default=None,
        required=False,
        help=(
            "The probability mask for gray matter voxels. Should approximately be in same space as the template."
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
    parser.add_argument(
        "--apriori_img_path",
        dest="apriori_img_path",
        default=None,
        required=False,
        help=(
            "Reduce the search space to the mask. If ``gm_probseg_img_path`` is supplied, the intersected group "
            "mask is restricted to the gray matter mask then restricted to the apriori mask. Note that "
            "cluster-based inference relies on spatially contiguous voxels, it is possible for clusters with the highest "
            "voxel statistical value to be in an apriori region and spread to non-apriori regions. This can result in the cluster  "
            "being deemed insignificant since the cluster extent can be reduced significantly. Recommend the apriori mask to be "
            "large regions (e.g., networks). Mask should approximately be in the same space as template."
        ),
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
        "--excluded_covariates",
        dest="excluded_covariates",
        default=["all"],
        required=False,
        nargs="*",
        type=str,
        help=(
            "Additional covariates to exclude from second level model. "
            "Should be a single string with variables separated as space or 'all' to exclude everything."
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
        type=float,
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
        type=float,
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


@dataclass
class DataContainer:
    columns_to_ignore: list[str] = field(
        default_factory=lambda: ["Subj", "session_id", "InputFile", "dose", "dose_mg"],
    )
    excluded_regressors: list = field(default_factory=list)
    categorical_regressors: set = field(
        default_factory=lambda: set(["sex", "race", "ethnicity"])
    )
    # From most to least
    deprioritized_regressors_order: list[str] = field(
        default_factory=lambda: [
            "ethnicity",
            "race",
            "sex",
            "age",
            "n_censored_volumes",
        ],
    )

    @staticmethod
    def get_glt_codes(cohort: str) -> str:
        glt_codes = {
            "kids": (
                "-gltCode 5_vs_0 'dose : 1*'5' -1*'0'' ",
                "-gltCode 10_vs_0 'dose : 1*'10' -1*'0'' ",
                "-gltCode 10_vs_5 'dose : 1*'10' -1*'5'' ",
                "-gltCode 0 'dose : 1*'0'' ",
                "-gltCode 5 'dose : 1*'5'' ",
                "-gltCode 10 'dose : 1*'10'' ",
                "-gltCode mean 'dose : {mean_code}' ",
            ),
            "adults": (
                "-gltCode mph_vs_placebo 'dose : 1*'mph' -1*'placebo' ",
                "-gltCode mph 'dose : 1*mph'' ",
                "-gltCode placebo 'dose : 1*'placebo'' ",
                "-gltCode mean 'dose : {mean_code}' ",
            ),
        }

        return glt_codes[cohort]

    def update_excluded_regressors(self, excluded_covariates: list[str]) -> None:
        excluded_covariates = [
            item for cov in excluded_covariates for item in cov.split() if cov
        ]
        if not excluded_covariates:
            return

        if "all" in excluded_covariates:
            self.excluded_regressors.extend(self.deprioritized_regressors_order)
            LGR.info(
                "Added the following variables to be excluded, if available: "
                f"{self.deprioritized_regressors_order}"
            )
        else:
            self.excluded_regressors.extend(excluded_covariates)
            LGR.info(
                "Added the following variables to be excluded, if available: "
                f"{excluded_covariates}"
            )

    @property
    def non_continuous_cols(self) -> list[str]:
        return self.columns_to_ignore + list(self.categorical_regressors)

    @property
    def exclude_afni_regressor_columns(self) -> list[str]:
        columns_to_skip = [col for col in self.columns_to_ignore if col != "dose"]

        return columns_to_skip + self.excluded_regressors


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


def create_data_table(bids_dir, datacontainer, subject_list, beta_files):
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
    # AFNI 26 requires first column to be named "Subj"
    data_table = data_table.rename(columns={"participant_id": "Subj"})

    for col in ["acq_time"]:
        if col in data_table.columns:
            data_table = data_table.drop(col, axis=1)

    column_names = (
        ["Subj", "dose"]
        + [
            name
            for name in data_table.columns
            if name not in ["Subj", "dose", "InputFile"]
        ]
        + ["InputFile"]
    )
    data_table = data_table.loc[:, column_names]
    data_table = data_table.dropna(how="all", axis=1).dropna(axis=0)
    if pd.to_numeric(data_table["dose"].dropna(), errors="coerce").notna().all():
        data_table["dose"] = data_table["dose"].astype(int).astype(str)
    else:
        data_table["dose"] = data_table["dose"].astype(str)

    if "dose_mg" in data_table.columns:
        data_table["dose_mg"] = data_table["dose_mg"].astype(int).astype(str)

    continuous_vars = set(data_table.columns).difference(
        datacontainer.non_continuous_cols + datacontainer.excluded_regressors
    )
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
    apriori_img_path,
    method,
    entity_key,
    first_level_glt_label,
    second_level_glt_code=None,
):
    if second_level_glt_code:
        group_mask_filename = (
            dst_dir
            / "group_masks"
            / f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{second_level_glt_code}_desc-{method}_group_mask.nii.gz"
        )
    else:
        group_mask_filename = (
            dst_dir
            / "group_masks"
            / f"task-{task}_{entity_key}-{first_level_glt_label}_desc-{method}_group_mask.nii.gz"
        )

    group_mask_filename.parent.mkdir(parents=True, exist_ok=True)

    subject_mask_files = []
    for filtered_beta_file in filtered_beta_files:
        sub_id = get_entity_value(filtered_beta_file, "sub")
        ses_id = get_entity_value(filtered_beta_file, "ses")

        kwargs = dict(
            scope="derivatives",
            subject=sub_id,
            task=task,
            suffix="mask",
            extension="nii.gz",
            return_type="file",
        )
        if ses_id:
            kwargs.update({"session": ses_id})

        mask_files = layout.get(**kwargs)

        mask_files = [mask_file for mask_file in mask_files if space in str(mask_file)]
        subject_mask_files.extend(mask_files)

    group_mask = intersect_masks(subject_mask_files, threshold=group_mask_threshold)
    group_mask = reduce_search_space(
        group_mask,
        gm_probseg_img_path,
        mask_type="gray_matter",
        gm_mask_threshold=gm_mask_threshold,
    )
    group_mask = reduce_search_space(group_mask, apriori_img_path, mask_type="apriori")

    nib.save(group_mask, group_mask_filename)

    return group_mask_filename


def reduce_search_space(group_mask, mask_img_path, mask_type, gm_mask_threshold=0.20):
    if not mask_img_path:
        return group_mask

    mask_img = nib.load(mask_img_path)
    resmapled_mask_img = resample_to_img(mask_img, group_mask, interpolation="nearest")

    if mask_type == "gray_matter":
        LGR.info(
            f"Thresholding gray metter probability mask using {Path(mask_img_path).name},  "
            f"any voxel with a probability greater than {gm_mask_threshold} "
            "will be retained in the group mask."
        )
        binarized_mask_image = resmapled_mask_img.get_fdata() > gm_mask_threshold
    else:
        binarized_mask_image = resmapled_mask_img.get_fdata() != 0

    group_mask_data = group_mask.get_fdata() * binarized_mask_image

    return nib.nifti1.Nifti1Image(group_mask_data, group_mask.affine, group_mask.header)


def get_glt_codes_str(data_table, datacontainer, cohort):
    glt_str = ""
    available_doses = sorted(data_table["dose"].astype(str).unique())
    for glt_code in datacontainer.get_glt_codes(cohort):
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


def get_model_str(data_table, datacontainer):
    columns = [
        col
        for col in data_table.columns
        if col not in datacontainer.exclude_afni_regressor_columns
    ]

    model_str = "+".join(set(columns))
    model_str += "+(1|Subj)"

    LGR.info(f"The following model will be used: {model_str}")

    return model_str


def get_centering_str(data_table, datacontainer):
    continuous_vars = set(data_table.columns).difference(
        datacontainer.non_continuous_cols + datacontainer.excluded_regressors
    )
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


def prioritize_regressors(design_matrix, datacontainer):
    has_dof = lambda design_matrix: design_matrix.shape[0] > design_matrix.shape[1]

    if has_dof(design_matrix):
        return design_matrix

    for regressor in datacontainer.deprioritized_regressors_order:
        if drop_columns := [
            col for col in design_matrix.columns if col.startswith(regressor)
        ]:
            LGR.info(f"Dropping the following regressor(s) to save dof: {drop_columns}")
            design_matrix = design_matrix.drop(columns=drop_columns)

        LGR.info(
            f"N OBSERVATIONS: {design_matrix.shape[0]}; N COLUMNS: {design_matrix.shape[0]}"
        )
        if has_dof(design_matrix):
            break

    return design_matrix


def get_col_from_data_table(data_table, datacontainer, col_type):
    if col_type == "categorical":
        cols = [
            col
            for col in data_table.columns
            if col in datacontainer.categorical_regressors
            and col
            not in datacontainer.excluded_regressors + datacontainer.columns_to_ignore
        ]
    else:
        cols = [
            col
            for col in data_table.columns
            if col
            not in datacontainer.non_continuous_cols + datacontainer.excluded_regressors
        ]

    return cols


def create_design_matrix(
    glt_data_table,
    datacontainer,
    include_intercept,
    average_within_subjects=False,
    second_level_glt_code=None,
):
    categorical_cols = get_col_from_data_table(
        glt_data_table, datacontainer, col_type="categorical"
    )
    continuous_cols = get_col_from_data_table(
        glt_data_table, datacontainer, col_type="continuous"
    )

    for col in categorical_cols:
        glt_data_table[col] = glt_data_table[col].astype(str)

    # Assumed to be the mean glt code
    if average_within_subjects:
        numeric_table = (
            glt_data_table.groupby("Subj")
            .mean(numeric_only=True)
            .reset_index(drop=True)
        )
        if "Subj" in numeric_table.columns:
            numeric_table = numeric_table.drop(columns=["Subj"])

        categorical_table = (
            glt_data_table.groupby("Subj")[categorical_cols]
            .first()
            .reset_index(drop=True)
        )
        if "Subj" in categorical_table.columns:
            categorical_table = categorical_table.drop(columns=["Subj"])

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
            glt_data_table[col], prefix=col, drop_first=True, dtype=int
        )
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
        design_matrix = prioritize_regressors(design_matrix, datacontainer)

    if include_intercept:
        # Using row shape of original table for case where design matrix is empty
        intercept = [1] * (glt_data_table.shape[0])
        design_matrix.insert(0, "intercept", intercept)
    else:
        first_label, _ = get_group_labels(second_level_glt_code)
        dose_codes = np.where(glt_data_table["dose"].astype(str) == first_label, 1, -1)
        design_matrix.insert(0, "dose", dose_codes)
        subject_regressors = pd.get_dummies(
            glt_data_table["Subj"], prefix="", prefix_sep="", dtype=int
        )
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
        first_label, second_label = get_group_labels(second_level_glt_code)
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
    datacontainer,
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
        datacontainer,
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
    datacontainer,
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

    glt_data_table = drop_within_subject_constant_regressors(
        datacontainer, glt_data_table
    )

    eb_data = glt_data_table["Subj"].factorize()[0] + 1
    np.savetxt(matrices_filenames_dict["eb_file"], eb_data, delimiter=",", fmt="%d")

    design_matrix = create_design_matrix(
        glt_data_table,
        datacontainer,
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


def set_permutations(max_permutation):
    n_permutations = min(max_permutation, 10000)

    LGR.info(f"Setting number of permutations to: {n_permutations}")

    return n_permutations


def compute_n_permutation(glt_data_table):
    n_subjects = len(glt_data_table["Subj"].unique())
    max_permutation = 2**n_subjects
    LGR.info(f"Maximum permutations possible = {max_permutation}")

    return set_permutations(max_permutation)


def compute_n_permutation_between_group(glt_data_table, cohort, second_level_glt_code):
    group_column = get_between_group_column(second_level_glt_code, cohort)
    first_label, _ = get_group_labels(second_level_glt_code)
    n1 = int((glt_data_table[group_column].astype(str) == first_label).sum())
    n2 = len(glt_data_table) - n1
    max_permutation = comb(n1 + n2, n1)
    LGR.info(f"Between-group permutations: ({n1}+{n2})-choose-{n1} = {max_permutation}")

    return set_permutations(max_permutation)


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
        for subject, group in glt_data_table.groupby("Subj"):
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


def create_difference_maps(
    data_table,
    output_dir,
    task,
    method,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
    afni_img_path=None,
    fsl_img_path=None,
    use_native_fsl=False,
):
    difference_dir = output_dir / "difference"
    difference_dir.mkdir(parents=True, exist_ok=True)

    diff_rows = []
    for subject, group in data_table.groupby("Subj"):
        mph_rows = group[group["dose"].astype(str) == "mph"]
        placebo_rows = group[group["dose"].astype(str) == "placebo"]

        if mph_rows.empty or placebo_rows.empty:
            LGR.warning(f"Subject {subject} missing mph or placebo visit, skipping.")
            continue

        mph_file = mph_rows["InputFile"].values[0]
        placebo_file = placebo_rows["InputFile"].values[0]

        prefix = (
            f"{subject}_task-{task}_{entity_key}-{first_level_glt_label}"
            f"_gltcode-{second_level_glt_code}"
        )

        diff_filename = (
            difference_dir / f"{prefix}_desc-{method}_difference_betas.nii.gz"
        )

        if afni_img_path:
            cmd = (
                f"apptainer exec -B /projects:/projects {afni_img_path} 3dcalc "
                f"-a {mph_file} -b {placebo_file} "
                f"-expr 'a-b' -prefix {diff_filename} -overwrite"
            )
        else:
            fsl_maths_call = (
                "fslmaths"
                if use_native_fsl
                else f"apptainer exec -B /projects:/projects {fsl_img_path} fslmaths"
            )
            cmd = f"{fsl_maths_call} {mph_file} -sub {placebo_file} {diff_filename}"

        LGR.info(f"Creating difference map for {subject}: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

        # Take the mph row as the template (has dose_mg) and update InputFile
        row = mph_rows.iloc[0].copy()
        row["InputFile"] = str(diff_filename)
        diff_rows.append(row)

    diff_table = pd.DataFrame(diff_rows).reset_index(drop=True)
    LGR.info(f"Created {len(diff_table)} difference maps for {second_level_glt_code}")

    diff_table.to_csv(
        output_dir
        / f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{second_level_glt_code}_desc-{method}_difference_data_table.tsv",
        sep="\t",
        index=None,
    )

    return diff_table


def drop_within_subject_constant_regressors(datacontainer, glt_data_table):
    remaining_columns = [
        col
        for col in glt_data_table.columns
        if col not in datacontainer.columns_to_ignore
    ]
    is_constant = lambda col: glt_data_table.groupby("Subj")[col].nunique().max() <= 1

    constant_columns = []
    for column in remaining_columns:
        if is_constant(column):
            constant_columns.append(column)

    if constant_columns:
        LGR.info(
            f"Removing the following columns that are constant within subjects: {constant_columns}"
        )

        for col in constant_columns:
            if col in glt_data_table.columns:
                glt_data_table = glt_data_table.drop(col, axis=1)

    return glt_data_table


def create_between_group_matrices(
    glt_data_table,
    datacontainer,
    output_dir,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
    cohort,
):
    matrices_filenames_dict = generate_matrices_filenames(
        output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
    )
    matrices_filenames_dict["eb_file"] = None

    group_column = get_between_group_column(second_level_glt_code, cohort)
    first_label, second_label = get_group_labels(second_level_glt_code)

    glt_data_table = drop_within_subject_constant_regressors(
        datacontainer, glt_data_table
    )

    group1_mask = (
        (glt_data_table[group_column].astype(str) == first_label).astype(int).to_numpy()
    )
    group2_mask = (
        (glt_data_table[group_column].astype(str) == second_label)
        .astype(int)
        .to_numpy()
    )

    design_components = {"Group1": group1_mask, "Group2": group2_mask}

    # Add covariates (continuous only, mean centered across all subjects)
    mean_center = lambda arr: arr - arr.mean()
    continuous_cols = get_col_from_data_table(
        glt_data_table, datacontainer, col_type="continuous"
    )
    for col in continuous_cols:
        design_components[col] = mean_center(glt_data_table[col].to_numpy())

    design_matrix = pd.DataFrame(design_components)
    design_matrix = prioritize_regressors(design_matrix, datacontainer)

    design_matrix.to_csv(
        matrices_filenames_dict["design_matrix_file"],
        sep=",",
        header=False,
        index=False,
    )
    create_header_file(design_matrix, matrices_filenames_dict["header_file"])

    # Contrast: [1 -1 0...] for Group1 > Group2, [-1 1 0...] for Group2 > Group1
    n_cols = design_matrix.shape[1]
    vector_pos = np.zeros(n_cols)
    vector_pos[0] = 1
    vector_pos[1] = -1
    vector_neg = vector_pos * -1

    np.savetxt(
        matrices_filenames_dict["contrast_matrix_file"],
        np.array([vector_pos, vector_neg]),
        delimiter=",",
        fmt="%.4f",
    )
    write_contrast_direction_file(
        output_dir, task, entity_key, first_level_glt_label, second_level_glt_code
    )

    return matrices_filenames_dict


def create_covariates_file_for_3dttest(
    glt_data_table,
    datacontainer,
    output_dir,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
):

    categorical_cols = get_col_from_data_table(
        glt_data_table, datacontainer, col_type="categorical"
    )
    covariates_cols = (
        get_col_from_data_table(glt_data_table, datacontainer, col_type="continuous")
        + categorical_cols
    )

    if not covariates_cols:
        return None

    prefix = (
        f"task-{task}_{entity_key}-{first_level_glt_label}"
        f"_gltcode-{second_level_glt_code}"
    )
    covariates_file = output_dir / f"{prefix}_desc-covariates.txt"

    cov_table = glt_data_table[["Subj"] + covariates_cols].copy()

    if categorical_cols:
        cov_table = pd.get_dummies(
            cov_table, columns=categorical_cols, drop_first=True, dtype=int
        )

    if cov_table.shape[1] < 2:
        return None

    cov_table["Subj"] = cov_table["Subj"].str.removeprefix("sub-")

    cov_table.to_csv(covariates_file, sep="\t", index=False)

    return covariates_file


def perform_3dttest(
    output_dir,
    task,
    entity_key,
    first_level_glt_label,
    second_level_glt_code,
    glt_data_table,
    group_mask_filename,
    afni_img_path,
    cohort,
    covariates_file=None,
):
    """
    Run 3dttest++ for between-group comparison on difference maps, only ever used for the difference maps.
    """
    output_filename = (
        output_dir / f"task-{task}_{entity_key}-{first_level_glt_label}"
        f"_gltcode-{second_level_glt_code}_desc-parametric_stats.nii.gz"
    )
    if output_filename.exists():
        output_filename.unlink()

    residual_filename = Path(str(output_filename).replace("_stats", "_residuals"))
    if residual_filename.exists():
        residual_filename.unlink()

    group_column = get_between_group_column(second_level_glt_code, cohort)
    first_label, second_label = get_group_labels(second_level_glt_code)

    # Build -setA and -setB strings
    set_a_rows = glt_data_table[glt_data_table[group_column] == first_label]
    set_b_rows = glt_data_table[glt_data_table[group_column] == second_label]

    set_a_str = " ".join(
        f"{row['Subj'].removeprefix('sub-')} {row['InputFile']}"
        for _, row in set_a_rows.iterrows()
    )
    set_b_str = " ".join(
        f"{row['Subj'].removeprefix('sub-')} {row['InputFile']}"
        for _, row in set_b_rows.iterrows()
    )

    covariates_str = (
        f"-covariates {covariates_file} -center SAME" if covariates_file else ""
    )

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dttest++ "
        f"-setA {first_label} {set_a_str} "
        f"-setB {second_label} {set_b_str} "
        f"-mask {group_mask_filename} "
        f"-prefix {output_filename} "
        f"-resid {residual_filename} "
        "-toz "
        f"{covariates_str}"
    )

    LGR.info(f"Running 3dttest++: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return residual_filename


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
    is_between_group,
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

    # The corrcon flag is used to control false positive rates since the positive and negative
    # maps are combined programatically later, which doubles the fpr (i.e., 2 × 0.05 = 0.10)
    # Note that Palm also has a two-tailed; however based on source code:
    #
    # https://github.com/andersonwinkler/PALM/blob/d937704489cc8ae8a7dafb273e421536f8b1e1a5/palm_core.m#L1603-L1605
    # if opts.twotail && ~ opts.missingdata && ( ~ opts.accel.lowrank || p > plm.nJ{m}(c))
    #      G{y}{m}{c} = abs(G{y}{m}{c});
    #   end
    #
    # The abs() is applied before tfce is computed:
    # https://github.com/andersonwinkler/PALM/blob/d937704489cc8ae8a7dafb273e421536f8b1e1a5/palm_core.m#L1674-L1675
    # if opts.tfce.uni.do
    #   tfce{y}{m}{c} = tfcefunc(G{y}{m}{c},y,opts,plm);
    #
    # tfce only integrates over positive thresholds:
    # https://github.com/andersonwinkler/PALM/blob/d937704489cc8ae8a7dafb273e421536f8b1e1a5/palm_tfce.m#L68-L75
    # for h = dh:dh:max(D(:));
    #    CC = bwconncomp(D>=h,opts.tfce.conn);
    #    integ = cellfun(@numel,CC.PixelIdxList).^opts.tfce.E * h^opts.tfce.H;
    #    for c = 1:CC.NumObjects,
    #        tfcestat(CC.PixelIdxList{c}) = ...
    #        tfcestat(CC.PixelIdxList{c}) + integ(c);
    #    end
    # end
    #
    # It appears that with -twotail, abs() makes all voxels positive before tfce, and
    # this would cause opposite effect directions (e.g., A > B and B > A) to be merged
    # into a single cluster which artificially inflates tfce. The intended behavior
    # for a true bidirectional map where each cluster only represents one direction is to do two separate
    # one-tailed contrasts with -corrcon, which corrects fwer across both contrasts, and merge the results
    # into a single bidirectional t-stat map.
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
        "-corrcon "
        "-logp "
        "-savedof "
        f"-o {output_prefix}"
    )

    if "_vs_" in second_level_glt_code and eb_file:
        palm_flags += f" -eb {eb_file} -ee -within"
    else:
        palm_flags += " -ise" if not is_between_group else " -ee"

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

    residual_log = Path(str(residual_filename).replace(".nii.gz", "_log.txt"))
    if residual_log.exists():
        residual_log.unlink()

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
    cohort,
    task,
    first_level_glt_label,
    analysis_type,
    space,
    gm_probseg_img_path,
    gm_mask_threshold,
    group_mask_threshold,
    apriori_img_path,
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
    excluded_covariates,
):
    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir) if deriv_dir else None
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)

    LGR.info(f"TASK: {task}, METHOD: {method}")

    if first_level_glt_label:
        first_level_glt_labels = (first_level_glt_label,)
    else:
        first_level_glt_labels = get_first_level_gltsym_codes(
            cohort, task, analysis_type, caller="second_level"
        )

    datacontainer = DataContainer()
    datacontainer.update_excluded_regressors(excluded_covariates)

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

        data_table = create_data_table(
            bids_dir, datacontainer, subject_list, beta_files
        )
        data_table_filename = (
            dst_dir
            / f"task-{task}_{entity_key}-{first_level_glt_label}_desc-data_table.txt"
        )
        data_table.to_csv(data_table_filename, sep="\t", index=False, encoding="utf-8")

        if method == "parametric":
            if not afni_img_path:
                LGR.warning("`afni_img_path` is required when method is parametric.")
                sys.exit(1)

            LGR.warning(
                f"Using {len(data_table['InputFile'].tolist())} files "
                f"from {len(data_table['Subj'].unique())} subjects for analysis "
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
                apriori_img_path,
                method,
                entity_key,
                first_level_glt_label,
            )

            glt_str = get_glt_codes_str(data_table, datacontainer, cohort)
            model_str = get_model_str(data_table, datacontainer)
            center_str = get_centering_str(data_table, datacontainer)

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

            # Run between-group codes (e.g., 15_vs_10) using 3dttest++
            between_group_codes = [
                code
                for code in get_second_level_glt_codes(cohort)
                if is_between_group_dose_code(code, cohort)
            ]
            for second_level_glt_code in between_group_codes:
                LGR.info(
                    f"Running between-group parametric analysis for: {second_level_glt_code}"
                )
                diff_output_dir = dst_dir / "second_level_outputs" / "parametric"
                diff_output_dir.mkdir(parents=True, exist_ok=True)

                diff_data_table = create_difference_maps(
                    data_table,
                    diff_output_dir,
                    task,
                    method,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                    afni_img_path=afni_img_path,
                )
                if diff_data_table is None or diff_data_table.empty:
                    LGR.warning(
                        f"No difference maps created for {second_level_glt_code}"
                    )
                    continue

                LGR.info(f"Creating group mask with threshold: {group_mask_threshold}")
                group_mask_filename = create_group_mask(
                    dst_dir,
                    get_layout(bids_dir, deriv_dir),
                    task,
                    space,
                    group_mask_threshold,
                    diff_data_table["InputFile"].tolist(),
                    gm_probseg_img_path,
                    gm_mask_threshold,
                    apriori_img_path,
                    method,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code=second_level_glt_code,
                )

                covariates_file = create_covariates_file_for_3dttest(
                    diff_data_table,
                    datacontainer,
                    diff_output_dir,
                    task,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                )
                between_group_residual = perform_3dttest(
                    diff_output_dir,
                    task,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                    diff_data_table,
                    group_mask_filename,
                    afni_img_path,
                    cohort,
                    covariates_file,
                )
                bg_acf = estimate_noise_smoothness(
                    dst_dir,
                    afni_img_path,
                    group_mask_filename,
                    between_group_residual,
                    first_level_glt_label,
                )
                perform_cluster_simulation(
                    afni_img_path,
                    group_mask_filename,
                    bg_acf,
                    first_level_glt_label,
                )
        else:
            # Nonparametric (PALM)
            use_native_palm = shutil.which("palm") is not None
            use_native_fsl = shutil.which("fslmerge") is not None

            if not fsl_img_path and not (use_native_palm or use_native_fsl):
                LGR.warning(
                    "`fsl_img_path` is required when method is nonparametric "
                    "and palm and `fslmerge` are not in path."
                )
                sys.exit(1)

            output_dir = dst_dir / "second_level_outputs" / "nonparametric"
            output_dir.mkdir(parents=True, exist_ok=True)
            for second_level_glt_code in get_second_level_glt_codes(cohort):
                LGR.info(f"Processing the following glt code: {second_level_glt_code}")

                is_between_group = is_between_group_dose_code(
                    second_level_glt_code, cohort
                )
                if not is_between_group:
                    vs_in_code = "_vs_" in second_level_glt_code
                    glt_data_table = drop_dose_rows(
                        data_table,
                        get_nontarget_dose(second_level_glt_code, cohort),
                        only_paired_data=vs_in_code,
                    )
                    glt_data_table = drop_constant_columns(glt_data_table)
                    if glt_data_table.empty:
                        LGR.info(
                            f"Skipping the following second level glt code: {second_level_glt_code}"
                        )
                        continue
                else:
                    # For between-group codes, create difference maps first
                    glt_data_table = create_difference_maps(
                        data_table,
                        output_dir,
                        task,
                        entity_key,
                        first_level_glt_label,
                        second_level_glt_code,
                        fsl_img_path=fsl_img_path,
                        use_native_fsl=use_native_fsl,
                    )
                    if glt_data_table is None or glt_data_table.empty:
                        LGR.warning(
                            f"No difference maps created for {second_level_glt_code}"
                        )
                        continue

                LGR.warning(
                    f"Using {len(glt_data_table['InputFile'].tolist())} files "
                    f"from {len(glt_data_table['Subj'].unique())} subjects for analysis "
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
                    apriori_img_path,
                    method,
                    entity_key,
                    first_level_glt_label,
                    second_level_glt_code,
                )

                if is_between_group:
                    matrix_creation_func = create_between_group_matrices
                    matrices_output_dict = matrix_creation_func(
                        glt_data_table,
                        datacontainer,
                        output_dir,
                        task,
                        entity_key,
                        first_level_glt_label,
                        second_level_glt_code,
                        cohort,
                    )
                    max_permutations = (
                        compute_n_permutation_between_group(
                            glt_data_table, cohort, second_level_glt_code
                        )
                        if n_permutations == "auto"
                        else n_permutations
                    )
                else:
                    matrix_creation_func = (
                        create_comparison_matrices
                        if vs_in_code
                        else create_mean_matrices
                    )
                    matrices_output_dict = matrix_creation_func(
                        glt_data_table,
                        datacontainer,
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
                    is_between_group,
                    max_permutations,
                    tfce_H,
                    tfce_E,
                    tfce_C,
                    use_native_palm,
                )
                for img in Path(output_prefix).parent.glob("*.nii"):
                    compress_image(
                        img, dst_dir=Path(output_prefix).parent, remove_src_file=True
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
