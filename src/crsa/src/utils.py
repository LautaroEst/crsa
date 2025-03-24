import sys
import yaml
import numpy as np

def save_yaml(data, file_path):
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=True)


def read_config_file(config_file):
    """
    Read the configuration file and return the configuration dictionary
    """
    with open(f"configs/{config_file}.yaml") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}
            
    return config


def is_list_of_strings(lst):
    """
    Check if the input is a list of strings
    """

    if not isinstance(lst, list):
        return False

    if not all(isinstance(x, str) for x in lst):
        return False
    
    return True

def is_numeric_ndarray(obj):
    return isinstance(obj, np.ndarray) and (obj.dtype.kind == 'f' or obj.dtype.kind == 'i') and np.isnan(obj).sum() == 0

def is_list_of_numbers(lst):
    return isinstance(lst, list) and all(isinstance(x, float) or isinstance(x, int) for x in lst)

def is_positive_number(obj):
    return isinstance(obj, float) or isinstance(obj, int) and obj > 0

def is_positive_integer(obj):
    return isinstance(obj, int) and obj > 0

def is_list_of_list_of_numbers(lst):
    return isinstance(lst, list) and all(is_list_of_numbers(x) for x in lst)