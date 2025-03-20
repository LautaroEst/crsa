
from ..src.utils import read_config_file

def main(
    depth = 2
):
    print(f"Running RSA with depth {depth}")


if __name__ == '__main__':
    config = read_config_file()
    main(**config)    
