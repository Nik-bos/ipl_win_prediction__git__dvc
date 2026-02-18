import pandas as pd
import argparse
from read_data import read_yaml
from sklearn.model_selection import train_test_split

def split_and_save(config)->None:
    yaml_data = read_yaml(config)
    
    train_data_path = yaml_data['split_data']['train_data_path']
    test_data_path = yaml_data['split_data']['test_data_path']
    test_size = yaml_data['split_data']['test_size']
    raw_data_path = yaml_data['load_data']['raw_dataset_path']
    raw_train_data_path = yaml_data['load_data']['raw_train_data']
    raw_test_data_path = yaml_data['load_data']['raw_test_data']

    random_state = yaml_data['base']['random_state']
    target_column = yaml_data["base"]['target_column']

    df = pd.read_csv(raw_data_path)
    print('Raw data successfully loaded.')
    print(f"Raw data shape: {df.shape}")
    print("="*50)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    train, test = train_test_split(
        df,
        test_size = test_size,
        random_state = random_state,
        shuffle = True,
        stratify = df[target_column]
    )

    print("Data successfully split into train and test")
    print(f"Train data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")
    print('='*50)

    print(f"""Train data class distribution:
{train[target_column].value_counts()}

Test data class distribution:
{test[target_column].value_counts()}
""")
    
    print("="*50)

    # Save train and test data
    train.to_csv(raw_train_data_path, index = False)
    test.to_csv(raw_test_data_path, index = False)

    print(f"""Train data saved to: {train_data_path}
Test data saved to: {test_data_path}
""")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "params.yaml")

    args = parser.parse_args()

    # Calling split_and_save function
    split_and_save(args.config)