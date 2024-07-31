"""Function to read value from configuration or use default."""


def get_config(config, *args, default=None, optional=True):
    """
    Function to get argument from config file and return default value if not found

    Inspired from hydromt.Model.get_config

    Parameters
    ----------
    config : dict
        config file
    args : tuple or string
        keys can given by multiple args: ('key1', 'key2')
        or a string with '.' indicating a new level: ('key1.key2')
    default : str/int/float/list, optional
        default value if argument not found, by default None
    optional : bool, optional
        if True, argument is optional, by default True
    """
    args = list(args)
    if len(args) == 1 and "." in args[0]:
        args = args[0].split(".") + args[1:]

    branch = config.copy()
    for key in args[:-1]:
        branch = branch.get(key, {})
        if not isinstance(branch, dict):
            branch = dict()
            break

    if args[-1] in branch:
        value = branch.get(args[-1])
        if value == "None":
            value = None
        return value
    elif optional:
        return default
    else:
        raise ValueError(f"Argument {args} not found in config file")
