import argparse, sys
from pathlib import Path

from nifti2bids.logging import setup_logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _utils import create_beta_files

LGR = setup_logger(__name__)


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
        help="Path to Apptainer image of Afni with R.",
    )
    parser.add_argument(
        "--subject",
        dest="subject",
        required=True,
        help="Subject ID without the 'sub-' entity.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--analysis_type",
        dest="analysis_type",
        required=True,
        choices=["glm", "gPPI"],
        help="The type of analysis performed (glm or gPPI).",
    )
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


def main(analysis_dir, subject, afni_img_path, task, analysis_type, out_dir):
    subject_base_dir = Path(analysis_dir) / (
        f"sub-{subject}" if not str(subject).startswith("sub-") else subject
    )
    sessions = [x.name for x in subject_base_dir.glob("*ses-*")]
    if not sessions:
        LGR.critical(f"No sessions for {subject} for {task}.")

    for session in sessions:
        subject_analysis_dir = subject_base_dir / session / "func"

        try:
            stats_file = next(
                subject_analysis_dir.glob(f"*task-{task}*desc-stats.nii.gz")
            )
        except Exception:
            LGR.critical(f"No stats file for subject {subject}, session {session}")
            continue

        beta_dir = subject_analysis_dir / "betas"
        beta_dir.mkdir(parents=True, exist_ok=True)

        create_beta_files(
            stats_file, beta_dir, afni_img_path, task, analysis_type, out_dir
        )


if __name__ == "__main__":
    _get_cmd_args = _get_cmd_args()
    args = _get_cmd_args.parse_args()
    main(**vars(args))
