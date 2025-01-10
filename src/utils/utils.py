# %%writefile ./utils/utils.py

import json

def read_json(file_path):
    """
    Reads a JSON file and returns the data as a Python object.

    :param file_path: Path to the JSON file
    :return: Parsed data from the JSON file
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Example usage
# data = read_json('data.json')
# print(data)










import os
import yaml
from omegaconf import OmegaConf

def load_args(args_file, verbose=False):
    """Load parameters from the params.yaml file."""
    with open(args_file, 'r') as file:
        # params = yaml.safe_load(file)
        args = OmegaConf.load(file)
        # args = OmegaConf.to_container(args, resolve=True)
    print(f"Loaded Args from {os.path.basename(args_file)}")
    
    if verbose:
        print(OmegaConf.to_yaml(args))
    # print("\n" + "=" * 40 + "\n")
    return args


import yaml

def update_yaml(file_path, key_path, new_value):
    """
    Updates a specific key in a YAML file with a new value.

    Args:
        file_path (str): Path to the YAML file.
        key_path (str): The hierarchical key path in the YAML file, e.g., "dataset.output_path".
        new_value (Any): The new value to assign to the specified key.

    Returns:
        Any: The updated value of the key.
    """
    try:
        # Load the YAML file
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Navigate to the nested key and update its value
        keys = key_path.split('.')
        current = data
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = new_value

        # Save the updated YAML file
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)

        return new_value

    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"Error: {e}")
        return None

# # Example usage
# yaml_file = "configs/prepare_data.yaml"
# updated_value = update_yaml_key(yaml_file, "dataset.output_path", "/kaggle/output/processed_data/")
# print(f"Updated dataset.output_path: {updated_value}")

    