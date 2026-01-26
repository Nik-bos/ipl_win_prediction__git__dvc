import argparse
from read_data import read_csv_file, read_yaml

def load_and_save_data(config):
    yaml_data = read_yaml(config)

    # We will need to hard code the path here as the path is not present in current working directory
    # This is work as expected in load_data.py in src
    df = read_csv_file(config) 

    # Save to data/raw
    raw_data_path = yaml_data["load_data"]['raw_dataset_path']
    df.to_csv(raw_data_path, index = False)
    print(f"Raw dataset saved to -> {raw_data_path}")
    print(f"Shape: {df.shape}")

# ==================================================
if __name__ == '__main__':
            parser = argparse.ArgumentParser()
            parser.add_argument("--config", default = 'params.yaml')
            args = parser.parse_args()

            load_and_save_data(args.config)
    