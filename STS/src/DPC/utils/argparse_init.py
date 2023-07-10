import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_non_default(parsed,parser):
    non_default = {
        opt.dest: getattr(parsed, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(parsed, opt.dest) and opt.default != getattr(parsed, opt.dest)
    }
