import argparse, shutil, subprocess, sys
from pathlib import Path

from nifti2bids.logging import setup_logger

LGR = setup_logger(__name__)
LGR.setLevel("INFO")


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Extract first level coefficient file from stats file."
    )
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help="Path to first level directory.",
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=True,
        help="Path to Singularity image of Afni with R",
    )
    parser.add_argument(
        "--subject",
        dest="subject",
        required=True,
        help="Subject ID without the 'sub-' entity.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--out_dir",
        dest="out_dir",
        required=False,
        default=None,
        help=(
            "Output directory to move contrast files to. "
            "If None, contrasts are saved in the analysis directory."
        ),
    )

    return parser


def _task_specific_contrasts(task):
    if task == "nback":
        contrasts = ("1-back_vs_0-back#0_Coef", "2-back_vs_0-back#0_Coef")
    elif task == "mtle":
        contrasts = "indoor#0_Coef"
    elif task == "mtlr":
        contrasts = "seen#0_Coef"
    elif task == "princess":
        contrasts = "switch_vs_nonswitch#0_Coef"
    else:
        contrasts = (
            "congruent_vs_neutral#0_Coef",
            "incongruent_vs_neutral#0_Coef",
            "nogo_vs_neutral#0_Coef",
            "congruent_vs_incongruent#0_Coef",
            "congruent_vs_nogo#0_Coef",
            "incongruent_vs_nogo#0_Coef",
        )

    return contrasts


def create_contrast_files(stats_file, contrast_dir, afni_path_img, task, out_dir):
    contrasts = _task_specific_contrasts(task)

    for contrast in contrasts:
        contrast_file = contrast_dir / stats_file.name.replace(
            "stats", contrast.replace("#0_Coef", "_betas")
        )
        cmd = (
            f"singularity exec -B /projects:/projects {afni_path_img} 3dbucket "
            f"{stats_file}'[{contrast}]' "
            f"-prefix {contrast_file} "
            "-overwrite"
        )
        LGR.info(f"Extracting {contrast} contrast: {cmd}")

        try:
            subprocess.run(cmd, shell=True, check=True)
        except Exception:
            LGR.critical(f"The following command failed: {cmd}", exc_info=True)

        if out_dir and contrast_file.exists():
            shutil.move(contrast_file, out_dir)


def main(analysis_dir, subject, afni_img_path, task, out_dir):
    subject_base_dir = Path(analysis_dir) / (
        f"sub-{subject}" if not str(subject).startswith("sub-") else subject
    )
    sessions = [x.name for x in subject_base_dir.glob("*ses-*")]
    if not sessions:
        LGR.critical(f"No sessions for {subject} for {task}.")

    for session in sessions:
        subject_analysis_dir = subject_base_dir / session / "func"
        stats_file = list(subject_analysis_dir.glob(f"*task-{task}*desc-stats.nii.gz"))
        if stats_file:
            stats_file = stats_file[0]
        else:
            sys.exit()

        contrast_dir = subject_analysis_dir / "contrasts"
        if not contrast_dir.exists():
            contrast_dir.mkdir()

        create_contrast_files(stats_file, contrast_dir, afni_img_path, task, out_dir)


if __name__ == "__main__":
    _get_cmd_args = _get_cmd_args()
    args = _get_cmd_args.parse_args()
    main(**vars(args))
