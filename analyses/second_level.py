import argparse, math, shutil, subprocess, sys
from functools import lru_cache
from pathlib import Path

import bids, nibabel as nib, numpy as np, pandas as pd
from nilearn.masking import intersect_masks

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger
from nifti2bids.qc import get_n_censored_volumes

from _utils import (
    get_contrast_entity_key,
    get_first_level_gltsym_codes,
    estimate_noise_smoothness,
    perform_cluster_simulation,
    threshold_palm_output,
)

LGR = setup_logger(__name__)

EXCLUDE_COLS = ["participant_id", "session_id", "InputFile", "dose"]
CATEGORICAL_VARS = set(["race", "ethnicity", "sex"])
SUBJECT_CONSTANT_VARS = ["age"] + list(CATEGORICAL_VARS)

GLT_CODES = (
    "-gltCode 5_vs_0 'dose : 1*'5' -1*'0'' ",
    "-gltCode 10_vs_0 'dose : 1*'10' -1*'0'' ",
    "-gltCode 10_vs_5 'dose : 1*'10' -1*'5'' ",
    "-gltCode mean 'dose : {mean_code}' ",
)


def _get_cmd_args():
    parser = argparse.ArgumentParser(description="Perform second level analysis.")
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
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--first_level_gltlabel",
        dest="first_level_gltlabel",
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
        "--mask_threshold",
        dest="mask_threshold",
        default=0.5,
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
        "--method",
        dest="method",
        default="non parametric",
        choices=["parametric", "nonparametric"],
        required=False,
        help=(
            "Whether to use 3dlmer (parametric) or Palm (nonparametric). "
            "Typically better to use nonparametric, it doesn't assume the erdistribution of the "
            "data and better controls false positives. If parametric is useror d then the "
            "acf method method should be used on the residuals to estimate smoothness and "
            "determine the appropriate cluster size via simulations. Nonparametric is stochastic "
            "however Palm sets seed to 0 by default for reproducibility. Though different results "
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
            "For 'auto' the permutation is computed by doing 1dose!^n * 2dose!^n * .... "
            "If the max permutation exceeds 10,000, then the permutation is set to 10,000. "
            "Lowest p-value -log10(1/1e4)."
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
        "--use_sign_flipping",
        dest="use_sign_flipping",
        default=False,
        required=False,
        help="If nonparametric, uses sign flipping for PALM",
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


def get_beta_files(analysis_dir, task, first_level_gltlabel):
    return sorted(
        list(Path(analysis_dir).rglob(f"*{task}*{first_level_gltlabel}*betas*.nii.gz"))
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
    data_table["dose"] = data_table["dose"].astype(int)

    exclude = list(CATEGORICAL_VARS) + EXCLUDE_COLS
    continuous_vars = set(data_table.columns).difference(exclude)
    for continuous_var in continuous_vars:
        data_table[continuous_var] = data_table[continuous_var].astype(float)

    nonconstant_columns = [
        col for col in data_table.columns if data_table[col].nunique() > 1
    ]
    data_table = data_table[nonconstant_columns]

    return data_table


@lru_cache()
def get_layout(bids_dir, deriv_dir):
    return bids.BIDSLayout(bids_dir, derivatives=deriv_dir or None)


def create_group_mask(layout, task, space, mask_threshold, beta_files):
    subject_mask_files = []
    for beta_file in beta_files:
        sub_id = get_entity_value(beta_file, "sub")
        ses_id = get_entity_value(beta_file, "ses")

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

    return intersect_masks(subject_mask_files, threshold=mask_threshold)


def get_glt_codes_str(data_table):
    glt_str = ""
    available_doses = sorted(data_table["dose"].unique())
    for glt_code in GLT_CODES:
        level_str = glt_code.removeprefix("-gltCode").lstrip().split(" ")[0]
        if level_str == "mean":
            value = round(1 / len(available_doses), 4)
            dose_list = [f"'{x}'" for x in available_doses]
            mean_code = f"{value}*" + f" +{value}*".join(dose_list)
            glt_str += glt_code.format(mean_code=mean_code)
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

    qvars_str = "'" + ",".join(continuous_vars) + "'"
    centers_str = "'" + ",".join(["0"] * len(continuous_vars)) + "'"

    centering_str = f"-qVars {qvars_str} -qVarCenters {centers_str}"

    LGR.info(f"The following centering string will be used: {centering_str}")

    return centering_str


def convert_table_to_matrices(
    data_table, dst_dir, task, entity_key, first_level_gltlabel
):
    """
    Takes the data table and creates matrices for PALM.

    For one-tailed tests, we create separate contrast matrices for positive
    and negative directions.

    Returns:
    - design_matrix_file: Design matrix CSV
    - eb_file: Exchangeability blocks file
    - beta_files_dict: Dict with "positive" and "negative" contrast matrix files
    - glt_codes_dict: Dict with "positive" and "negative" glt code lists
    """
    design_matrix_file = (
        dst_dir
        / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-design_matrix.csv"
    )
    eb_file = (
        dst_dir
        / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-exchangeability_blocks.csv"
    )
    contrast_matrix_file_pos = (
        dst_dir
        / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-contrast_matrix_pos.csv"
    )
    contrast_matrix_file_neg = (
        dst_dir
        / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-contrast_matrix_neg.csv"
    )

    header_file = (
        dst_dir
        / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-header_names.txt"
    )

    eb_data = data_table["participant_id"].factorize()[0] + 1
    LGR.info(f"Saving eb file to: {eb_file}")
    np.savetxt(eb_file, eb_data, delimiter=",", fmt="%d")

    available_doses = list(map(int, sorted(data_table["dose"].unique())))
    LGR.info(f"Available doses: {available_doses}")

    dose_dummies = pd.get_dummies(data_table["dose"], prefix="dose").astype(int)
    design_components = [dose_dummies]

    # Including subject regressors in fixed effects model so drop columns that are constant within subjects
    for col in SUBJECT_CONSTANT_VARS:
        if col in data_table.columns:
            data_table = data_table.drop(col, axis=1)

    categorical_cols = []

    continuous_cols = [
        col
        for col in data_table.columns
        if col not in EXCLUDE_COLS and col not in categorical_cols
    ]

    if continuous_cols:
        covariates = data_table[continuous_cols]
        for col in continuous_cols:
            covariates[col] = covariates[col] - covariates[col].mean()

        design_components.append(covariates)

    if categorical_cols:
        for col in categorical_cols:
            # Dropping first of other categorical columns to avoid
            # linear dependency
            dummies = pd.get_dummies(
                data_table[col], prefix=col, drop_first=True
            ).astype(int)
            design_components.append(dummies)

    # Create intercept for each subject except the first one to account for
    # within subject variance for repeat design
    subject_regressors = pd.get_dummies(
        data_table["participant_id"], prefix="", prefix_sep="", drop_first=True
    ).astype(int)
    design_components.append(subject_regressors)

    design_matrix = pd.concat(design_components, axis=1)

    with open(header_file, "w") as f:
        f.write(f"{','.join(design_matrix.columns.tolist())}")

    LGR.info(f"Design matrix columns: {design_matrix.columns.tolist()}")
    LGR.info(f"Saving design matrix file to: {design_matrix_file}")

    design_matrix.to_csv(design_matrix_file, sep=",", header=False, index=False)

    # Build contrasts for both directions
    contrasts_pos = []
    contrasts_neg = []
    glt_codes_pos = []
    glt_codes_neg = []

    dose_to_col = {dose: index for index, dose in enumerate(available_doses)}

    for index, dose_high in enumerate(available_doses):
        for dose_low in available_doses[:index]:
            # Positive direction: dose_high > dose_low (e.g., 5_vs_0)
            vector_pos = np.zeros(design_matrix.shape[1])
            vector_pos[dose_to_col[dose_high]] = 1
            vector_pos[dose_to_col[dose_low]] = -1
            contrasts_pos.append(vector_pos)
            glt_codes_pos.append(f"{dose_high}_vs_{dose_low}")

            # Negative direction: dose_low > dose_high (e.g., 0_vs_5)
            vector_neg = np.zeros(design_matrix.shape[1])
            vector_neg[dose_to_col[dose_low]] = 1
            vector_neg[dose_to_col[dose_high]] = -1
            contrasts_neg.append(vector_neg)
            glt_codes_neg.append(f"{dose_low}_vs_{dose_high}")

    # Add contrast for mean > 0 and < 0
    dose_cols_indices = list(dose_to_col.values())

    vector_pos = np.zeros(design_matrix.shape[1])
    vector_pos[dose_cols_indices] = round(1 / len(available_doses), 4)
    contrasts_pos.append(vector_pos)
    glt_codes_pos.append(f"mean")

    # Zeroes will show up as negative, that is fine -0 == 0 returns True
    vector_neg = vector_pos * -1
    contrasts_neg.append(vector_neg)
    glt_codes_neg.append(f"mean")

    contrast_matrix_pos = np.array(contrasts_pos)
    contrast_matrix_neg = np.array(contrasts_neg)

    LGR.info(f"Positive contrast names: {glt_codes_pos}")
    LGR.info(f"Saving positive contrast matrix file to: {contrast_matrix_file_pos}")
    np.savetxt(contrast_matrix_file_pos, contrast_matrix_pos, delimiter=",", fmt="%.4f")

    LGR.info(f"Negative contrast names: {glt_codes_neg}")
    LGR.info(f"Saving negative contrast matrix file to: {contrast_matrix_file_neg}")
    np.savetxt(contrast_matrix_file_neg, contrast_matrix_neg, delimiter=",", fmt="%.4f")

    # Save glt codes for reference
    glt_codes_file = (
        dst_dir / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-glt_codes.txt"
    )
    with open(glt_codes_file, "w") as f:
        f.write("# Positive direction contrasts:\n")
        for i, name in enumerate(glt_codes_pos, 1):
            f.write(f"c{i}_pos: {name.replace('_vs_', ' > ')}\n")

        f.write("\n# Negative direction contrasts:\n")
        for i, name in enumerate(glt_codes_neg, 1):
            f.write(f"c{i}_neg: {name.replace('_vs_', ' > ')}\n")

    beta_files_dict = {
        "positive": contrast_matrix_file_pos,
        "negative": contrast_matrix_file_neg,
    }
    glt_codes_dict = {
        "positive": glt_codes_pos,
        "negative": glt_codes_neg,
    }

    return design_matrix_file, eb_file, beta_files_dict, glt_codes_dict


def compute_n_permutation(data_table, use_sign_flipping):
    # TODO: Possibly put in utils and add (n1 + n2)!/(n1! * n2!) or 2^N option
    factor = 2 if use_sign_flipping else 1
    n_available_doses_counts = (
        data_table["participant_id"].value_counts().value_counts()
    )
    counts_list = list(
        zip(n_available_doses_counts.index, n_available_doses_counts.values)
    )
    products = [
        math.factorial(int(k)) ** (int(n_subjects)) * factor ** (int(n_subjects))
        for k, n_subjects in counts_list
    ]
    product = math.prod(products)
    LGR.info(f"Maximum permutations possible: {product}")

    n_permutations = min(product, 10000)
    LGR.info(f"Setting number of permutations to: {n_permutations}")

    return n_permutations


def perform_palm(
    dst_dir,
    beta_files,
    group_mask_filename,
    design_matrix_file,
    eb_file,
    contrast_matrix_files_dict,
    fsl_img_path,
    task,
    entity_key,
    first_level_gltlabel,
    n_permutations,
    tfce_H,
    tfce_E,
    tfce_C,
    use_native_palm,
    use_native_fsl,
    use_sign_flipping,
):
    concatenated_filename = (
        dst_dir
        / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-group_concatenated.nii.gz"
    )

    if concatenated_filename.exists():
        concatenated_filename.unlink()

    fsl_call = (
        "fslmerge"
        if use_native_fsl
        else f"apptainer exec -B /projects:/projects {fsl_img_path} fslmerge"
    )
    cmd = (
        f"{fsl_call} "
        f"-t {concatenated_filename} "
        f"{' '.join([str(f) for f in beta_files])}"
    )
    LGR.info(f"Concatenating images: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    base_dir = dst_dir / "palm_outputs"
    base_dir.mkdir(parents=True, exist_ok=True)

    output_prefixes = {}
    for direction in ["positive", "negative"]:
        output_prefix = (
            base_dir
            / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-nonparametric_{direction}"
        )
        contrast_matrix_file = contrast_matrix_files_dict[direction]

        palm_flags = (
            "-noniiclass "
            f"-i {concatenated_filename} "
            f"-m {group_mask_filename} "
            f"-d {design_matrix_file} "
            f"-t {contrast_matrix_file} "
            f"-eb {eb_file} "
            "-ee "
            "-within "
            f"-n {n_permutations} "
            "-T "
            f"-tfce_H {tfce_H} "
            f"-tfce_E {tfce_E} "
            f"-tfce_C {tfce_C} "
            "-logp "
            "-savedof "
            f"-o {output_prefix}"
        )

        if use_sign_flipping:
            palm_flags += " -ise"

        if use_native_palm:
            palm_dir = Path(shutil.which("palm")).parent
            cmd = f"matlab -nodisplay -nosplash -r \"addpath('{palm_dir}'); palm {palm_flags}; exit;\""
        else:
            cmd = (
                f"apptainer exec -B /projects:/projects {fsl_img_path} "
                f"octave --eval 'palm {palm_flags}'"
            )

        LGR.info(f"Running PALM ({direction} direction): {cmd}")
        subprocess.run(cmd, shell=True, check=True)

        output_prefixes[direction] = output_prefix

    return output_prefixes


def perform_3dlmer(
    task,
    entity_key,
    first_level_gltlabel,
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
        / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-parametric_stats.nii.gz"
    )
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    if output_filename.exists():
        LGR.info("Replacing stats file")
        output_filename.unlink()

    residual_filename = Path(str(output_filename).replace("_stats", "_residuals"))
    if residual_filename.exists():
        LGR.info("Replacing residual file")
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
    first_level_gltlabel,
    analysis_type,
    space,
    mask_threshold,
    afni_img_path,
    fsl_img_path,
    method,
    n_permutations,
    tfce_H,
    tfce_E,
    tfce_C,
    use_sign_flipping,
    cluster_correction_p,
    n_cores,
    exclude_niftis_file,
):
    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir) if deriv_dir else None
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)

    LGR.info(f"TASK: {task}, METHOD: {method}")

    if first_level_gltlabel:
        first_level_gltlabels = [first_level_gltlabel]
    else:
        first_level_gltlabels = get_first_level_gltsym_codes(
            task, analysis_type, caller="second_level"
        )

    for first_level_gltlabel in first_level_gltlabels:
        entity_key = get_contrast_entity_key(first_level_gltlabel)
        LGR.info(f"FIRST LEVEL GLTLABEL: {first_level_gltlabel}")
        beta_files = exclude_beta_files(
            get_beta_files(analysis_dir, task, first_level_gltlabel),
            exclude_niftis_file,
        )

        if not beta_files:
            LGR.warning(f"No beta files found for {first_level_gltlabel}")
            continue

        subject_list = get_subjects(beta_files)
        LGR.info(
            f"Found {len(beta_files)} files from {len(set(subject_list))} subjects"
        )

        LGR.info("Creating data table.")
        data_table = create_data_table(bids_dir, subject_list, beta_files)

        data_table_filename = (
            dst_dir
            / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-data_table.txt"
        )

        # Get the beta files only in the table, these are the participants used
        filtered_beta_files = data_table["InputFile"].to_numpy(copy=True).tolist()

        LGR.critical(
            f"Using {len(filtered_beta_files)} files from {len(data_table['participant_id'].unique())} subjects for analysis"
        )

        LGR.info(f"Creating group mask with threshold: {mask_threshold}")
        group_mask = create_group_mask(
            get_layout(bids_dir, deriv_dir),
            task,
            space,
            mask_threshold,
            filtered_beta_files,
        )
        group_mask_filename = (
            dst_dir
            / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-group_mask.nii.gz"
        )
        LGR.info(f"Saving group mask to: {group_mask_filename}")
        nib.save(group_mask, group_mask_filename)

        if method == "parametric":
            if not afni_img_path:
                LGR.critical("afni_img_path is required when method is parametric.")
                sys.exit(1)

            data_table["dose"] = data_table["dose"].astype(str)
            LGR.info(f"Saving data table to: {data_table_filename}")
            data_table.to_csv(
                data_table_filename, sep="\t", index=False, encoding="utf-8"
            )

            glt_str = get_glt_codes_str(data_table)
            model_str = get_model_str(data_table)
            center_str = get_centering_str(data_table)

            residual_filename = perform_3dlmer(
                task,
                entity_key,
                first_level_gltlabel,
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
                first_level_gltlabel,
            )

            perform_cluster_simulation(
                dst_dir,
                afni_img_path,
                group_mask_filename,
                acf_parameters_filename,
                first_level_gltlabel,
            )
        else:
            # Nonparametric (PALM)
            use_native_palm = shutil.which("palm") is not None
            use_native_fsl = shutil.which("fslmerge") is not None

            if not fsl_img_path and not (use_native_palm or use_native_fsl):
                LGR.critical(
                    "fsl_img_path is required when method is nonparametric and palm and fslmerge are not in path."
                )
                sys.exit(1)

            LGR.info(f"Saving data table to: {data_table_filename}")
            data_table.to_csv(data_table_filename, sep=" ", index=False)

            design_matrix_file, eb_file, contrast_matrix_files_dict, glt_codes_dict = (
                convert_table_to_matrices(
                    data_table,
                    dst_dir,
                    task,
                    entity_key,
                    first_level_gltlabel,
                )
            )

            if n_permutations == "auto":
                n_permutations = compute_n_permutation(data_table, use_sign_flipping)

            output_prefixes = perform_palm(
                dst_dir,
                filtered_beta_files,
                group_mask_filename,
                design_matrix_file,
                eb_file,
                contrast_matrix_files_dict,
                fsl_img_path,
                task,
                entity_key,
                first_level_gltlabel,
                n_permutations,
                tfce_H,
                tfce_E,
                tfce_C,
                use_native_palm,
                use_native_fsl,
                use_sign_flipping,
            )

            threshold_palm_output(
                output_prefixes,
                glt_codes_dict,
                cluster_correction_p,
                dst_dir=dst_dir,
            )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
