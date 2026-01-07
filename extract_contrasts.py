import argparse
import logging
import subprocess
import sys
from pathlib import Path

try:
    sys.path.append("/code")
    from utils import setup_logger

    LGR = setup_logger("FirstLevel")
    LGR.setLevel("INFO")
except ImportError:
    LGR = logging.getLogger("FirstLevel")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel("INFO")

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    LGR.addHandler(console_handler)


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

    return parser


def _task_specific_contrasts(task):
    if task == "nback":
        contrasts = ("1-back_vs_0-back#0_Coef", "2-back_vs_0-back#0_Coef")
    elif task == "mtle":
        pass
    elif task == "mtlr":
        pass
    elif task == "princess":
        pass
    else:
        pass

    return contrasts


def create_contrast_files(stats_file, contrast_dir, afni_path_img, task):
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
        subprocess.run(cmd, shell=True, check=True)


def main(analysis_dir, subject, afni_img_path, task):
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

        create_contrast_files(stats_file, contrast_dir, afni_img_path, task)


if __name__ == "__main__":
    _get_cmd_args = _get_cmd_args()
    args = _get_cmd_args.parse_args()
    main(**vars(args))
