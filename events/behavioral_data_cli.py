from _events_utils import _get_cmd_args

from get_behavioral_data import run_pipeline

if __name__ == "__main__":
    cmd_args = _get_cmd_args(caller="Behavioral Data")
    args = cmd_args.parse_args()
    run_pipeline(**vars(args), caller="Behavioral Data")
