import argparse


def n_dummy_type(x):
    if x == "auto":
        return "auto"
    else:
        try:
            x = int(x)

            return x
        except ValueError:
            raise argparse.ArgumentTypeError(
                "`n_dummy_scans` must be 'auto' or an integer."
            )
