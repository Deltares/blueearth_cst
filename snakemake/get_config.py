"""Function to read value from configuration or use default."""


def get_config(config, arg, default=None, optional=True):
    """
    Function to get argument from config file and return default value if not found

    Parameters
    ----------
    config : dict
        config file
    arg : str
        argument to get from config file
    default : str/int/float/list, optional
        default value if argument not found, by default None
    optional : bool, optional
        if True, argument is optional, by default True
    """
    if arg in config:
        value = config[arg]
        if value == "None":
            value = None
        return value
    elif optional:
        return default
    else:
        raise ValueError(f"Argument {arg} not found in config file")
