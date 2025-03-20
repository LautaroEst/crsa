import sys
import yaml


def read_config_file():
    """
    Read the configuration file and return the configuration dictionary
    """

    if len(sys.argv) < 2:
        raise ValueError("Please provide a configuration file")

    config_file = sys.argv[1]

    with open(f"configs/{config_file}.yaml") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}
            
    return config