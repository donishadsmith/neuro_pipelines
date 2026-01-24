import argparse, subprocess
from functools import lru_cache
from pathlib import Path

import bids, pandas as pd, nibabel as nib
from nilearn.masking import intersect_masks

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger

from _utils import get_task_contrasts

LGR = setup_logger(__name__)
LGR.setLevel("INFO")


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
    # Doing contrasts as opposed to BRIK selection to prevent indexing errors
    # in the event subject has the stats BRIK but not a specific contrast.
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
        help="Value between 0 to 1 denoting the level of intersection for the masks.",
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=True,
        help="Path to Singularity image of Afni with R.",
    )
    parser.add_argument(
        "--n_cores",
        dest="n_cores",
        default=1,
        required=False,
        help="Number of cores to use.",
    )
    parser.add_argument(
        "--exclude_niftis_file",
        dest="exclude_niftis_file",
        default=None,
        required=False,
        help=(
            "Prefixes of the filename of the NIfTI images to exclude not the full filename. "
            "Entities included should be 'sub', 'ses', 'task', and 'run' "
            "(i.e. 'sub-01_ses-01_task-nback_run-01' not "
            "'sub-01_ses-01_task-nback_run-01_desc_bold.nii.gz'). Should contain a single "
            "column named 'nifti_prefix_filename' Files excluded should be determined using MRIQC or "
            "other factors such as participant falling asleep during task."
        ),
    )

    return parser


def get_contrast_files(contrast_dir, task, contrast):
    return sorted(list(Path(contrast_dir).rglob(f"*{task}*{contrast}*.nii.gz")))


def filter_contrasts_files(contrast_files, exclude_niftis_file):
    if not exclude_niftis_file:
        return contrast_files

    df = pd.read_csv(exclude_niftis_file, sep=None, engine="python")
    exlcuded_niftis_prefixes = [
        Path(nifti_prefix_filename).name.split("_desc")[0]
        for nifti_prefix_filename in df["nifti_prefix_filename"].to_numpy()
    ]

    return [
        contrast_file
        for contrast_file in contrast_files
        if Path(contrast_file).name.split("_space")[0] not in exlcuded_niftis_prefixes
    ]


def get_subjects(contrast_files):
    # Get the available subjects from the contrasts
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

        sessions_dfs.append(df)

    all_sessions = pd.concat(sessions_dfs, ignore_index=True)
    data_table = all_sessions.merge(participants_df, on="participant_id")
    column_names = (
        ["participant_id"]
        + [
            name
            for name in data_table.columns
            if name not in ["participant_id", "InputFile"]
        ]
        + ["InputFile"]
    )
    data_table = data_table.loc[:, column_names]
    data_table = data_table.dropna()

    data_table["dose"] = data_table["dose"].astype(int).astype(str)

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


def perform_3dlmer(
    task,
    contrast,
    dst_dir,
    data_table_filename,
    group_mask_filename,
    afni_img_path,
    n_cores,
    glt_str,
):
    output_filename = dst_dir / f"task-{task}_contrast-{contrast}_desc-stats.nii.gz"
    residual_filename = str(output_filename).replace("-stats", "-residuals")

    cmd = (
        f"singularity exec -B /projects:/projects {afni_img_path} 3dLMEr "
        f"-mask {group_mask_filename} "
        "-model 'dose+age+(1|participant_id)' "
        f"-jobs {n_cores} "
        "-qVars 'age' "
        "-qVarCenters '0' "
        "-dbgArgs "
        f"{glt_str}"
        f"-prefix {output_filename} "
        f"-resid {residual_filename} "
        f"-dataTable @{data_table_filename}"
    )

    LGR.info(f"Running 3dLMEr: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main(
    bids_dir,
    deriv_dir,
    contrast_dir,
    dst_dir,
    task,
    space,
    mask_threshold,
    afni_img_path,
    n_cores,
    exclude_niftis_file,
):
    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)
    contrast_dir = Path(contrast_dir)
    dst_dir = Path(dst_dir)

    LGR.info(f"TASK: {task}")

    contrasts = get_task_contrasts(task, caller="second_level")
    for contrast in contrasts:
        LGR.info(f"CONTRAST: {contrast}")
        contrast_files = filter_contrasts_files(
            get_contrast_files(contrast_dir, task, contrast), exclude_niftis_file
        )
        subject_list = get_subjects(contrast_files)

        LGR.info("Creating datatable.")
        data_table = create_data_table(bids_dir, subject_list, contrast_files)
        glt_str = get_glt_codes_str(data_table)

        data_table_filename = (
            dst_dir / f"task-{task}_contrast-{contrast}_desc-data_table.txt"
        )
        LGR.info(f"Saving datatable to: {data_table_filename}")
        data_table.to_csv(data_table_filename, sep=" ", index=False)

        LGR.info(f"Creating group mask with the current threshold: {mask_threshold}")
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

        perform_3dlmer(
            task,
            contrast,
            dst_dir,
            data_table_filename,
            group_mask_filename,
            afni_img_path,
            n_cores,
            glt_str,
        )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
