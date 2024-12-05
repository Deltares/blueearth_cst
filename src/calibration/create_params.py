import json
from itertools import product
from pathlib import Path
import pandas as pd
import numpy as np


def create_set(p: Path | str):
    """
    This function takes a path to a JSON file as input, reads the file, and creates a pandas DataFrame from the data.
    The DataFrame is created by taking the Cartesian product of the "values" lists from the JSON data, and using the "short_name" fields as column names.
    The function also returns a list of the keys in the JSON data, and a list of the "method" fields from the JSON data.

    Parameters:
    p (Path | str): The path to the JSON file to read.

    Returns:
    lnames (list): A list of the keys in the JSON data.
    methods (list): A list of the "method" fields from the JSON data.
    ds (DataFrame): A pandas DataFrame created from the JSON data.
    """
    with open(p, "r") as _r:
        data = json.load(_r)

    # Extract the cartesian product of all values
    params = list(product(*[item["values"] for item in data.values()]))

    # Extract column names, handling comma-separated short_names
    columns = []
    split_dict = {}
    for item in data.values():
        if ',' in item["short_name"]:
            split = item["short_name"].split(',')
            split_dict[split[0]] = split[1:]
            columns.append(split[0])
        else:
            columns.append(item["short_name"])
    
    # Create the DataFrame
    ds = pd.DataFrame(params, columns=columns)
   
    if len(split_dict) > 0:
        for col in ds.columns:
            if col in split_dict:
                for i in split_dict[col]:
                    ds[i] = ds[col]   

    # Extract the keys as lnames, handling comma-separated keys
    lnames = []
    for key in data.keys():
        lnames.extend(key.split(','))

    # Extract methods, handling cases where a method needs to be duplicated for split keys
    methods = []
    for item in data.values():
        method_count = len(item["short_name"].split(','))
        methods.extend([item["method"]] * method_count)
    
    #wflow vars
    wflow_vars = []
    for item in data.values():
        wflow_var_count = len(item["wflow"].split(','))
        wflow_vars.extend([item["wflow"]] * wflow_var_count)

    return lnames, methods, ds, wflow_vars


if __name__ == "__main__":
    fn = R"N:\My Documents\unix\documents\RWSoS\RWSOS_Calibration\meuse\config\MINIMAL_calib_recipe.json"
    lnames, methods, ds = create_set(fn)
    print(ds)