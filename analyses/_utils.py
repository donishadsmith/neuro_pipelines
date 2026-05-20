"Shared utilities"

import base64, shutil, subprocess
from pathlib import Path

import nibabel as nib, numpy as np, pandas as pd
from nilearn.image import resample_to_img

from bidsaid.logging import setup_logger
from bidsaid.metadata import needs_resampling

LGR = setup_logger(__name__)

VALID_TASK_NAMES = {
    "kids": ["nback", "mtlr", "mtle", "flanker", "princess"],
    "adults": ["nback", "mtlr", "mtle", "flanker", "simplegng", "complexgng"],
}

TASK_CONTRASTS = {
    "kids": {
        "nback": (
            "1-back_vs_center",
            "2-back_vs_center",
            "2-back_vs_1-back",
        ),
        "mtle": ("neutral_encoding",),
        "mtlr": ("neutral_retrieval",),
        "princess": ("switch_vs_nonswitch",),
        "flanker": (
            "incongruent_vs_congruent",
            "nogo_vs_neutral",
            "nogo",
            "incongruent",
            "congruent",
        ),
    },
    "adults": {
        "nback": ("2-back_vs_0-back",),
        "mtle": (
            "aversive_encoding_vs_neutral_encoding",
            "neutral_encoding",
        ),
        "mtlr": (
            "aversive_retrieval_vs_neutral_retrieval",
            "neutral_retrieval",
        ),
        "flanker": (
            "incongruent_vs_congruent",
            "nogo_vs_neutral",
            "nogo",
            "incongruent",
            "congruent",
        ),
        "simplegng": ("nogo_vs_go",),
        "complexgng": ("nogo_vs_go",),
    },
}

# For the kids data, its fully within crossover design so [10 - placebo] - [5 - placebo]) = 10 - 5
CONTRAST_CODES = {
    "kids": ("5_vs_0", "10_vs_0", "10_vs_5", "mph_vs_placebo", "mean"),
    "adults": ("mph_vs_placebo", "mean"),
}


def get_first_level_gltsym_codes(cohort, task, analysis_type, caller):
    contrasts = TASK_CONTRASTS[cohort][task]
    if analysis_type == "gPPI":
        contrasts = modify_contrast_names(contrasts)

    return (
        tuple(f"{contrast}#0_Coef" for contrast in contrasts)
        if caller == "extract_betas"
        else contrasts
    )


def get_second_level_glt_codes(cohort):
    return CONTRAST_CODES[cohort]


def modify_contrast_names(contrasts):
    modified_contrasts = []
    for contrast in contrasts:
        if "_vs_" in contrast:
            modified_contrasts.append(
                f"PPI_{contrast.split('_vs_')[0]}_vs_PPI_{contrast.split('_vs_')[1]}"
            )

        else:
            modified_contrasts.append(f"PPI_{contrast}")

    return modified_contrasts


def get_beta_names(gltsyms, add_coef_str=False, create_sub_conditions=True):
    if isinstance(gltsyms, str):
        gltsyms = [gltsyms]

    if not create_sub_conditions:
        return gltsyms

    beta_names = []
    for gltsym in gltsyms:
        names = [] if "_vs_" not in gltsym else gltsym.split("_vs_")
        names += [gltsym]

        beta_names.extend(names)

    if add_coef_str:
        beta_names = [
            f"{name}#0_Coef" if not name.endswith("#0_Coef") else name
            for name in beta_names
        ]

    return list(set(beta_names))


def get_contrast_name_from_file(filename):
    filename = Path(filename).name

    return filename.split("desc-")[-1].split("_betas")[0]


def get_contrast_entity_key(input_str):
    input_str = Path(input_str).name

    return "contrast" if "_vs_" in input_str else "condition"


def resample_seed_img(seed_img, subject_nifti_img):
    if needs_resampling(seed_img, subject_nifti_img):
        seed_img = resample_to_img(
            seed_img, subject_nifti_img, interpolation="nearest", copy_header=True
        )

    return seed_img


def get_coordinate_from_filename(filepath, replace_underscore=True):
    filepath = Path(filepath)
    possible_coordinate = ""

    for marker in ("_sphere_mask_", "_cluster_mask_", "_individual_betas_"):
        if marker in filepath.name:
            possible_coordinate = filepath.name.split(marker)[1]
            suffix = "".join(filepath.suffixes[3:])
            possible_coordinate = possible_coordinate.removesuffix(suffix)
            if replace_underscore:
                possible_coordinate = possible_coordinate.replace("_", ",")

            break

    return possible_coordinate


def create_beta_files(
    stats_file,
    beta_dir,
    afni_img_path,
    cohort,
    task,
    analysis_type,
    out_dir=None,
    overwrite=True,
):
    first_level_gltsyms = get_first_level_gltsym_codes(
        cohort, task, analysis_type, caller="extract_betas"
    )
    beta_names = get_beta_names(first_level_gltsyms, add_coef_str=True)

    for beta_name in beta_names:
        beta_file = beta_dir / stats_file.name.replace(
            "stats", beta_name.replace("#0_Coef", "_betas")
        )
        if beta_file.exists() and overwrite:
            beta_file.unlink()

        cmd = (
            f"apptainer exec -B /projects:/projects {afni_img_path} 3dbucket "
            f"{stats_file}'[{beta_name}]' "
            f"-prefix {beta_file} "
            "-overwrite"
        )
        LGR.info(f"Extracting {beta_name} betas: {cmd}")

        try:
            subprocess.run(cmd, shell=True, check=True)
        except Exception:
            LGR.critical(f"The following command failed: {cmd}", exc_info=True)

        if out_dir and beta_file.exists():
            path = Path(out_dir) / beta_file.name
            if path.exists():
                LGR.info("Replacing old file with new file.")
                path.unlink()

            shutil.move(beta_file, out_dir)


def get_nontarget_dose(second_level_glt_code, cohort):
    if second_level_glt_code == "mean":
        return []

    # For kids mph_vs_placebo, all three original doses (0, 5, 10) are needed
    # since 5 and 10 are averaged into a single mph image at the second level
    if cohort == "kids" and second_level_glt_code == "mph_vs_placebo":
        return []

    doses = {"kids": {"0", "5", "10"}, "adults": {"mph", "placebo"}}

    return list(doses[cohort].difference(second_level_glt_code.split("_vs_")))


def needs_complete_cases(method):
    return False if method == "parametric" else True


def drop_dose_rows(
    data_table,
    nontarget_dose_list,
    only_complete_cases=False,
    return_removed_subjects=False,
):
    if not nontarget_dose_list and only_complete_cases is False:
        return data_table

    removed_subjects = []
    target_doses = (
        data_table.loc[
            ~data_table["dose"].astype(str).isin(nontarget_dose_list), "dose"
        ]
        .unique()
        .tolist()
    )
    target_doses = list(map(str, target_doses))
    data_table = data_table[data_table["dose"].astype(str).isin(target_doses)]
    if only_complete_cases:
        counts = data_table["Subj"].value_counts()
        removed_subjects = counts[counts < len(target_doses)].index.tolist()
        data_table = data_table[~data_table["Subj"].isin(removed_subjects)]
        total_subjects = len(data_table["Subj"].unique())
        contrast_name = get_contrast_name_from_file(data_table["InputFile"].tolist()[0])

        LGR.warning(
            f"For contrast ({contrast_name}), the following subjects have been removed: {removed_subjects}. "
            f"A total of {total_subjects} unique subjects with the following doses remain: {target_doses}"
        )

    if return_removed_subjects:
        return data_table, removed_subjects
    else:
        return data_table


def get_group_labels(second_level_glt_code):
    return second_level_glt_code.split("_vs_")


def save_binary_mask(mask_img_fdata, affine, hdr, mask_filename):
    """To save as a true binary mask and prevent equality index due to floating point issues"""
    mask_img_fdata = mask_img_fdata.astype(np.int8)
    hdr.set_data_dtype(np.int8)

    mask_img = nib.nifti1.Nifti1Image(mask_img_fdata, affine, hdr)

    nib.save(mask_img, mask_filename)


# pd.read_csv(exclude_niftis_files, sep=None, engine="python") fails in cases
# where there is only one column and row
def _get_dataframe(filename):
    try:
        return pd.read_excel(filename)
    except:
        pass

    df = pd.read_csv(filename, sep=None, engine="python")
    if "nifti_prefix_filename" not in df.columns:
        # Any separator will work
        df = pd.read_csv(filename)
        if "nifti_prefix_filename" not in df.columns:
            raise Exception(
                "`exclude_niftis_file` must contain a column named 'nifti_prefix_filename'."
            )

    return df


def delete_dir(dirname):
    """
    Delete dir to prevent file pollution due to re-running pipeline. Differences in number
    of clusters could occur so not all files will be overwritten. Use only for the "get_cluster_results"
    and "extract_individual_betas" pipelines. These are were this issue occurs and were the pipeline is not
    expected to run in parallel and write to the same output directory like "second_level.py". Also
    used for the "first_level" pipeline to clean out directory due to changes in exclusion criteria
    """
    if Path(dirname).exists():
        shutil.rmtree(dirname, ignore_errors=True)


def create_condition_label_str(beta_name):
    if "_vs_" in beta_name:
        first_condition_label, second_condition_label = beta_name.split("_vs_")
        condition_label = f"{first_condition_label} - {second_condition_label}"
    else:
        condition_label = f"{beta_name} only"

    return condition_label


def embed_image(image_path):
    """
    Reads the bytes from the image, then converts to base64,
    its binary-to-text encoding that uses 64 printable characters
    to represent each 6-bit segment of a sequence of byte values
    (https://en.wikipedia.org/wiki/Base64)

    It allows an image to be embedded in an html file, which will be
    completely self-contained and won't the file path to the image.
    """
    data = Path(image_path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")

    return f"data:image/png;base64,{b64}"
