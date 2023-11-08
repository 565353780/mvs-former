from functools import reduce
from operator import getitem


def get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)


def set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(";")
    get_by_path(tree, keys[:-1])[keys[-1]] = value


# helper functions to update config dict with custom cli options
def update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            set_by_path(config, k, v)
    return config
