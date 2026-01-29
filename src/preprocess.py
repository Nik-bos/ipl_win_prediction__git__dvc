from read_data import read_yaml


import pandas as pd
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer

from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler

import joblib
import argparse

import warnings
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)



# Display settings
pd.set_option("display.max_columns", None)
sklearn.set_config(transform_output = 'pandas') # To display sklearn outputs as pandas DataFrames

# ========================= Read raw csv file =========================

def read_raw_csv(df_path: str)-> pd.DataFrame:
    # yaml_data = read_yaml(params_path)
    # raw_dataset_path = yaml_data['load_data']['raw_dataset_path']
    
    df = pd.read_csv(df_path)
    return df

# df = read_raw_csv(params_path= "params.yaml")
# print(df.head())

# Cleaning Steps
# 1) Changing `Royal Challengers **Bangalore**` to `Royal Challengers **Bengaluru**`
# 2) In city also **Bangalore** to **Bengaluru**

# ========================= Initial Cleaning =========================


def initial_cleaning(df: pd.DataFrame)-> pd.DataFrame:
    change = {
        'Bangalore': 'Bengaluru',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'}

    df[['batting_team', 'bowling_team', 'city']] = df[['batting_team', 'bowling_team', 'city']].replace(change)

    teams = ['Rising Pune Supergiant', "Rising Pune Supergiants", "Gujarat Lions", "Pune Warriors", "Kochi Tuskers Kerala"]

    df = df[~
        (
            df['batting_team'].isin(teams) |
            df['bowling_team'].isin(teams)
            )
    ]

    # Balls left > 125 changing to 120
    df['balls_left'] = np.where(df['balls_left']>120, 120, df['balls_left'])

    print('Data cleaned successfully')

    return df

# ========================= Column-wise Cleaning =========================

def column_wise_cleaning()->ColumnTransformer:
    ohe = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
    outliers_capping = Winsorizer(capping_method='iqr')
    scaler = MinMaxScaler()
    pt = PowerTransformer(standardize = True)

    batting_bowling_city_steps = [
        ('ohe', ohe)
    ]
    batting_bowling_city_pipeline = Pipeline(steps = batting_bowling_city_steps)

    runs_balls_left_steps = [
        ("outliers_capping", outliers_capping),
        ('scaling', scaler)
    ]

    runs_balls_left_pipeline = Pipeline(steps = runs_balls_left_steps)

    wickets_remaining_steps = [
    ('scaling', scaler)
    ]

    wickets_remaining_pipeline = Pipeline(wickets_remaining_steps)

    total_run_x_steps = [
    ("outliers_capping", Winsorizer(capping_method="iqr")),
    ('scaling', MinMaxScaler())
    ]

    total_run_x_pipeline = Pipeline(total_run_x_steps)

    crr_rrr_steps = [
        ('outliers_capping', outliers_capping),
        ('PowerTransformer', pt)
    ]

    crr_rrr_pipeline = Pipeline(steps = crr_rrr_steps)

    transformers = [
        ('Trans_1', batting_bowling_city_pipeline, ['batting_team', 'bowling_team', 'city']),
        ('Trans_2', runs_balls_left_pipeline, ['runs_left', 'balls_left']),
        ('Trans_3', wickets_remaining_pipeline, ['wickets_remaining']),
        ("Trans_4", total_run_x_pipeline, ['total_run_x']),
        ('Trans_5', crr_rrr_pipeline, ['crr', 'rrr'])
    ]

    ct_pipeline = ColumnTransformer(
        transformers = transformers,
        remainder = 'passthrough',
        verbose_feature_names_out=False # prevents double prefixing
        )
    
    # ColumnTransformer prefixes the transformer name (batting_team) to the OHE’s own naming convention (batting_team_Chennai Super Kings).
    # Result → batting_team__batting_team_Chennai Super Kings.
    # So we used, verbose_feature_names_out - False

    return ct_pipeline

# ========================= Final Pipeline =========================



def create_final_pipeline()-> Pipeline:
    
    final_pipeline_steps = [
        ('Initial_cleaning', FunctionTransformer(func = initial_cleaning)),
        ("Column_transformer", column_wise_cleaning())
    ]

    final_pipeline = Pipeline(steps = final_pipeline_steps)

    return final_pipeline

# final_pipeline = create_final_pipeline()
# trans_df = final_pipeline.fit_transform(df)

#==================================================

# Saving pipeline and preprocessed data
def save_pipeline_and_trans_data(config):
    yaml_data = read_yaml(config)
    raw_dataset_path = yaml_data['load_data']['raw_dataset_path']
    preprocessed_data_path = yaml_data['split_data']['trans_data_path']
    pipeline_path = yaml_data['pipeline']['pipeline_path']

    df_raw = read_raw_csv(raw_dataset_path)
    # print(df_raw.columns)
    final_pipeline = create_final_pipeline()

    trans_df = final_pipeline.fit_transform(df_raw)

    joblib.dump(final_pipeline, pipeline_path)
    print(f"Preprocessor pipeline saved to-> {pipeline_path}")

    # Saving transformed data
    trans_df.to_csv(preprocessed_data_path, index = False)
    print(f"Transformed data saved to-> {preprocessed_data_path}")


#==================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "params.yaml")

    args = parser.parse_args()
    save_pipeline_and_trans_data(args.config)
