import argparse, subprocess, sys
from functools import lru_cache
from pathlib import Path

import bids, nibabel as nib, numpy as np, pandas as pd
from nilearn.masking import intersect_masks
from nilearn.image import new_img_like

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger

from _utils import (
    get_task_contrasts,
    estimate_noise_smoothness,
    perform_cluster_simulation,
    get_number_of_censored_volumes,
)

LGR = setup_logger(__name__)
LGR.setLevel("INFO")

EXCLUDE_COLS = ["participant_id", "session_id", "InputFile", "dose"]
CATEGORICAL_VARS = set(["race", "ethnicity", "sex"])


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
        "--contrast_dir",
        dest="contrast_dir",
        required=True,
        help=(
            "Path to directory containing the extracted contrasts. "
            "Contrasts are grabbed recursively, only the naming of the contrasts "
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
        help=(
            "Path to Apptainer image of Afni with R. " "Required if using parametric."
        ),
    )
    parser.add_argument(
        "--fsl_img_path",
        dest="fsl_img_path",
        required=False,
        default=None,
        help=(
            "Path to apptainer image of FSL with Palm using Octave. "
            "Required if method is nonparametric."
        ),
    )
    parser.add_argument(
        "--method",
        dest="method",
        default="parametric",
        choices=["parametric", "nonparametric"],
        required=False,
        help=(
            "Whether to use 3dlmer (parametric) or Palm (nonparametric). "
            "Typically better to use nonparametric, it doesn't assume the distribution of the "
            "data and better controls false positives. If parametric is used then the "
            "acf method method should be used on the residuals to estimate smoothness and "
            "determine the appropriate cluster size via simulations."
        ),
    )
    parser.add_argument(
        "--n_permutations",
        dest="n_permutations",
        default=10000,
        type=int,
        required=False,
        help="If method is nonparametric, the number of permutations to pass to Palm.",
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


def get_contrast_files(contrast_dir, task, contrast):
    return sorted(list(Path(contrast_dir).rglob(f"*{task}*{contrast}*.nii.gz")))


def filter_contrasts_files(contrast_files, exclude_niftis_file):
    if not exclude_niftis_file:
        return contrast_files

    df = pd.read_csv(exclude_niftis_file, sep=None, engine="python")
    excluded_niftis_prefixes = [
        Path(nifti_prefix_filename).name.split("_desc")[0]
        for nifti_prefix_filename in df["nifti_prefix_filename"].tolist()
    ]

    return [
        contrast_file
        for contrast_file in contrast_files
        if Path(contrast_file).name.split("_space")[0] not in excluded_niftis_prefixes
    ]


def get_subjects(contrast_files):
    return sorted([get_entity_value(file, "sub") for file in contrast_files])


def create_data_table(bids_dir, subject_list, contrast_files):
    bids_dir = Path(bids_dir)
    participants_df = pd.read_csv(bids_dir / "participants.tsv", sep="\t")

    session_files = sorted(list(bids_dir.rglob("sub-*_sessions.tsv")))
    sessions_dfs = []
    for session_file in session_files:
        if (sub_id := get_entity_value(session_file, "sub")) not in subject_list:
            continue

        df = pd.read_csv(session_file, sep="\t")
        df["participant_id"] = f"sub-{sub_id}"

        subject_contrast_files = [
            str(file) for file in contrast_files if sub_id in str(file)
        ]
        for subject_contrast_file in subject_contrast_files:
            ses_id = get_entity_value(
                subject_contrast_file, "ses", return_entity_prefix=True
            )
            df.loc[df["session_id"] == ses_id, "InputFile"] = subject_contrast_file
            censor_file = ""
            while not censor_file.name == "func":
                if censor_file == censor_file.parents[-1]:
                    break
                else:
                    censor_file = censor_file / "censor.1D"

            if censor_file:
                df.loc[df["session_id"] == ses_id, "n_censored_volumes"] = (
                    get_number_of_censored_volumes(censor_file)
                )
            else:
                df.loc[df["session_id"] == ses_id, "n_censored_volumes"] = np.nan

        sessions_dfs.append(df)

    all_sessions = pd.concat(sessions_dfs, ignore_index=True)
    data_table = all_sessions.merge(participants_df, on="participant_id")

    if "acq_date" in data_table.columns:
        data_table = data_table.drop("acq_date", axis=1)

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

    return data_table


@lru_cache()
def get_layout(bids_dir, deriv_dir):
    return bids.BIDSLayout(bids_dir, derivatives=deriv_dir or None)


def create_group_mask(layout, task, space, mask_threshold, contrast_files):
    subject_mask_files = []
    for contrast_file in contrast_files:
        sub_id = get_entity_value(contrast_file, "sub")
        ses_id = get_entity_value(contrast_file, "ses")

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
    glt_codes = (
        "-gltCode 5_vs_0 'dose : 1*'5' -1*'0'' ",
        "-gltCode 10_vs_0 'dose : 1*'10' -1*'0'' ",
        "-gltCode 10_vs_5 'dose : 1*'10' -1*'5'' ",
    )

    glt_str = ""
    available_doses = data_table["dose"].unique()
    for glt_code in glt_codes:
        level_str = glt_code.removeprefix("-gltCode").lstrip().split(" ")[0]
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
    quoted_vars = [f"'{var}'" for var in continuous_vars]
    quoted_zeroes = ["'0'"] * len(continuous_vars)

    centering_str = (
        f"-qVars {' '.join(quoted_vars)}  -qVarCenters {' '.join(quoted_zeroes)} "
    )

    return centering_str


def convert_table_to_matrices(data_table, dst_dir, task, contrast):
    """
    Takes the data table and creates matrices for PALM.

    For one-tailed tests, we create separate contrast matrices for positive
    and negative directions.

    Returns:
    - design_matrix_file: Design matrix CSV
    - eb_file: Exchangeability blocks file
    - contrast_files_dict: Dict with "positive" and "negative" contrast matrix files
    - glt_codes_dict: Dict with "positive" and "negative" glt code lists
    """
    design_matrix_file = (
        dst_dir / f"task-{task}_contrast-{contrast}_desc-design_matrix.csv"
    )
    eb_file = (
        dst_dir / f"task-{task}_contrast-{contrast}_desc-exchangeability_blocks.csv"
    )
    contrast_matrix_file_pos = (
        dst_dir / f"task-{task}_contrast-{contrast}_desc-contrast_matrix_pos.csv"
    )
    contrast_matrix_file_neg = (
        dst_dir / f"task-{task}_contrast-{contrast}_desc-contrast_matrix_neg.csv"
    )

    eb_data = data_table["participant_id"].factorize()[0] + 1
    LGR.info(f"Saving eb file to: {eb_file}")
    np.savetxt(eb_file, eb_data, delimiter=",", fmt="%d")

    available_doses = sorted(data_table["dose"].unique())
    LGR.info(f"Available doses: {available_doses}")

    dose_dummies = pd.get_dummies(data_table["dose"], prefix="dose").astype(int)
    design_components = [dose_dummies]

    categorical_cols = list(
        set(["race", "education", "sex"]).intersection(data_table.columns.tolist())
    )

    continuous_cols = [
        col
        for col in data_table.columns
        if col not in EXCLUDE_COLS and col not in categorical_cols
    ]

    if continuous_cols:
        covariates = data_table[continuous_cols].copy()
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

    design_matrix = pd.concat(design_components, axis=1)

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

    contrast_matrix_pos = np.array(contrasts_pos)
    contrast_matrix_neg = np.array(contrasts_neg)

    LGR.info(f"Positive contrast names: {glt_codes_pos}")
    LGR.info(f"Saving positive contrast matrix file to: {contrast_matrix_file_pos}")
    np.savetxt(contrast_matrix_file_pos, contrast_matrix_pos, delimiter=",", fmt="%d")

    LGR.info(f"Negative contrast names: {glt_codes_neg}")
    LGR.info(f"Saving negative contrast matrix file to: {contrast_matrix_file_neg}")
    np.savetxt(contrast_matrix_file_neg, contrast_matrix_neg, delimiter=",", fmt="%d")

    # Save glt codes for reference
    glt_codes_file = dst_dir / f"task-{task}_contrast-{contrast}_desc-glt_codes.txt"
    with open(glt_codes_file, "w") as f:
        f.write("# Positive direction contrasts:\n")
        for i, name in enumerate(glt_codes_pos, 1):
            f.write(f"c{i}_pos: {name}\n")
        f.write("\n# Negative direction contrasts:\n")
        for i, name in enumerate(glt_codes_neg, 1):
            f.write(f"c{i}_neg: {name}\n")

    contrast_files_dict = {
        "positive": contrast_matrix_file_pos,
        "negative": contrast_matrix_file_neg,
    }
    glt_codes_dict = {
        "positive": glt_codes_pos,
        "negative": glt_codes_neg,
    }

    return design_matrix_file, eb_file, contrast_files_dict, glt_codes_dict


def perform_palm(
    dst_dir,
    contrast_files,
    group_mask_filename,
    design_matrix_file,
    eb_file,
    contrast_matrix_files_dict,
    fsl_img_path,
    task,
    contrast,
    n_permutations,
):
    concatenated_filename = (
        dst_dir / f"task-{task}_contrast-{contrast}_desc-concatenated.nii.gz"
    )

    if concatenated_filename.exists():
        concatenated_filename.unlink()

    cmd = (
        f"apptainer exec -B /projects:/projects {fsl_img_path} fslmerge "
        f"-t {concatenated_filename} "
        f"{' '.join([str(f) for f in contrast_files])}"
    )
    LGR.info(f"Concatenating images: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    output_prefixes = {}
    for direction in ["positive", "negative"]:
        output_prefix = (
            dst_dir / f"task-{task}_contrast-{contrast}_desc-nonparametric_{direction}"
        )
        contrast_matrix_file = contrast_matrix_files_dict[direction]

        cmd = (
            f"apptainer run -B /projects:/projects {fsl_img_path} "
            "octave eval 'palm "
            "-noniiclass "
            f"-i {concatenated_filename} "
            f"-m {group_mask_filename} "
            f"-d {design_matrix_file} "
            f"-t {contrast_matrix_file} "
            f"-eb {eb_file} "
            "-ise "
            "-within "
            f"-n {n_permutations} "
            "-T "
            "-tfce_C 6 "
            "-logp "
            "-savedof "
            f"-o {output_prefix}'"
        )

        LGR.info(f"Running PALM ({direction} direction): {cmd}")
        subprocess.run(cmd, shell=True, check=True)

        output_prefixes[direction] = output_prefix

    return output_prefixes


def threshold_palm_output(
    output_prefixes, glt_codes_dict, cluster_correction_p, dst_dir
):
    logp_threshold = -np.log10(cluster_correction_p)
    LGR.info(
        f"Using -log10(p) threshold: {logp_threshold:.4f} "
        f"(cluster_significance={cluster_correction_p})"
    )

    for direction, prefix_path in output_prefixes.items():
        glt_codes = glt_codes_dict[direction]
        output_dir = prefix_path.parent
        prefix = prefix_path.name

        # If only one contrast, palm excludes c{index}; however
        # a minimum of two contrasts are needed since only one tail will
        # be used
        for index, glt_code in enumerate(glt_codes, 1):
            LGR.info(f"Processing {direction} contrast {index}: {glt_code}")

            tstat_file = output_dir / f"{prefix}_tfce_tstat_c{index}.nii.gz"
            pval_file = output_dir / f"{prefix}_tfce_tstat_fwep_c{index}.nii.gz"

            if not pval_file.exists():
                LGR.warning(f"Missing file: {pval_file}")
                continue

            tstat_img = nib.load(tstat_file)
            pval_img = nib.load(pval_file)

            sig_mask = (pval_img.get_fdata() > logp_threshold).astype(float)
            masked_tstat = tstat_img.get_fdata() * sig_mask
            thresholded_img = new_img_like(
                tstat_img, masked_tstat, affine=tstat_img.affine, copy_header=True
            )

            # Use glt_code in filename (e.g., 5_vs_0 or 0_vs_5)
            thresholded_file = (
                dst_dir / f"task-{prefix.split('task-')[1].split('_contrast')[0]}_"
                f"contrast-{prefix.split('contrast-')[1].split('_desc')[0]}_"
                f"gltcode-{glt_code}_desc-nonparametric_cluster_corrected.nii.gz"
            )
            nib.save(thresholded_img, thresholded_file)

            LGR.info(f"Saved thresholded t-map: {thresholded_file}")


def perform_3dlmer(
    task,
    contrast,
    dst_dir,
    data_table_filename,
    group_mask_filename,
    afni_img_path,
    model_str,
    center_str,
    glt_str,
    n_cores,
):
    output_filename = dst_dir / f"task-{task}_contrast-{contrast}_desc-stats.nii.gz"
    if output_filename.exists():
        LGR.info("Replacing stats file")
        output_filename.unlink()

    residual_filename = Path(str(output_filename).replace("-stats", "-residuals"))
    if residual_filename.exists():
        LGR.info("Replacing residual file")
        residual_filename.unlink()

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dLMEr "
        f"-mask {group_mask_filename} "
        f"-model '{model_str}' "
        f"-jobs {n_cores} "
        f"{center_str}"
        "-dbgArgs "
        f"{glt_str}"
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
    contrast_dir,
    dst_dir,
    task,
    space,
    mask_threshold,
    afni_img_path,
    fsl_img_path,
    method,
    n_permutations,
    cluster_correction_p,
    n_cores,
    exclude_niftis_file,
):
    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir) if deriv_dir else None
    contrast_dir = Path(contrast_dir)
    dst_dir = Path(dst_dir)

    LGR.info(f"TASK: {task}, METHOD: {method}")

    contrasts = get_task_contrasts(task, caller="second_level")
    for contrast in contrasts:
        LGR.info(f"CONTRAST: {contrast}")
        contrast_files = filter_contrasts_files(
            get_contrast_files(contrast_dir, task, contrast), exclude_niftis_file
        )

        if not contrast_files:
            LGR.warning(f"No contrast files found for {contrast}")
            continue

        subject_list = get_subjects(contrast_files)
        LGR.info(f"Found {len(contrast_files)} files from {len(subject_list)} subjects")

        LGR.info(f"Creating group mask with threshold: {mask_threshold}")
        group_mask = create_group_mask(
            get_layout(bids_dir, deriv_dir),
            task,
            space,
            mask_threshold,
            contrast_files,
        )
        group_mask_filename = (
            dst_dir / f"task-{task}_contrast-{contrast}_desc-group_mask.nii.gz"
        )
        LGR.info(f"Saving group mask to: {group_mask_filename}")
        nib.save(group_mask, group_mask_filename)

        LGR.info("Creating data table.")
        data_table = create_data_table(bids_dir, subject_list, contrast_files)

        data_table_filename = (
            dst_dir / f"task-{task}_contrast-{contrast}_desc-data_table.txt"
        )

        if method == "parametric":
            if not afni_img_path:
                LGR.critical("afni_img_path is required when method is parametric.")
                sys.exit(1)

            data_table["dose"] = data_table["dose"].astype(str)
            LGR.info(f"Saving data table to: {data_table_filename}")
            data_table.to_csv(data_table_filename, sep=" ", index=False)

            glt_str = get_glt_codes_str(data_table)
            model_str = get_model_str(data_table)
            center_str = get_centering_str(data_table)

            residual_filename = perform_3dlmer(
                task,
                contrast,
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
                contrast,
            )

            perform_cluster_simulation(
                dst_dir,
                afni_img_path,
                group_mask_filename,
                acf_parameters_filename,
                contrast,
            )
        else:
            # Nonparametric (PALM)
            if not fsl_img_path:
                LGR.critical("fsl_img_path is required when method is nonparametric.")
                sys.exit(1)

            LGR.info(f"Saving data table to: {data_table_filename}")
            data_table.to_csv(data_table_filename, sep=" ", index=False)

            design_matrix_file, eb_file, contrast_matrix_files_dict, glt_codes_dict = (
                convert_table_to_matrices(data_table, dst_dir, task, contrast)
            )

            output_prefixes = perform_palm(
                dst_dir,
                contrast_files,
                group_mask_filename,
                design_matrix_file,
                eb_file,
                contrast_matrix_files_dict,
                fsl_img_path,
                task,
                contrast,
                n_permutations,
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
