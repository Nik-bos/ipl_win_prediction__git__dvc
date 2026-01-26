# Read parameters.yaml   (To get the path of original csv file)
# Read original csv file
# Return a dataframe

import yaml
import pandas as pd
import argparse

# Function to read params.yaml

def read_yaml(config):
    with open(config, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    return yaml_data

#===============================

# Function to read and csv file

def read_csv_file(config):

    # Reading yaml file
    yaml_data = read_yaml(config = config)
    orig_data_path = yaml_data["data_source"]["original_data_path"]    
    df = pd.read_csv(orig_data_path, index_col=0)

    print("Original dataset read successfully")
    print(f"Shape: {df.shape}")
    print("="*50)
    return df

#===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = 'params.yaml')

    args = parser.parse_args()

    df = read_csv_file(args.config)
    print(df.head())
