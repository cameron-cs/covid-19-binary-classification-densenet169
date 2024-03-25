import json


def load_json_config(config_path):
    """Load the JSON configuration file."""
    with open(config_path, 'r') as config_file:
        return json.load(config_file)
