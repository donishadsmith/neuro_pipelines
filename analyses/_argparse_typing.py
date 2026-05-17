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


def boolean_flags(x):
    return True if x.lower() in ["t", "true", "1", "y", "yes"] else False


def convert_none(cast=str):
    def _convert(x):
        if str(x).lower() == "none":
            return None

        return cast(x)

    return _convert
