from _events_utils import _get_cmd_args

from create_event_files import run_pipeline

if __name__ == "__main__":
    cmd_args = _get_cmd_args(caller="BIDS Events")
    args = cmd_args.parse_args()
    run_pipeline(**vars(args), caller="BIDS Events")
