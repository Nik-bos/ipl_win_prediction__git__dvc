from read_data import read_yaml
import os
import joblib
import json
import argparse

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, accuracy_score, confusion_matrix, classification_report

# Always show numbers without scientific notation (Pandas)
pd.set_option('display.float_format', '{:.3f}'.format)

# Disable scientific notation globally (Numpy)
np.set_printoptions(suppress=True, precision=3)

# ==================== Evaluate Model ====================

def evaluate_model(actual, predicted):

    # Finding precision, recall and threshold
    precision, recall, threshold = precision_recall_curve(actual, predicted[:, 1])

    # Calculating f1_scores
    f1_scores = (2* (precision*recall)/ (precision+recall+1e-8))
    # Finding best threshold
    best_threshold = threshold[np.argmax(f1_scores)]

    pred = np.where(predicted[:, 1]> best_threshold, 1, 0)

    acc_score = accuracy_score(actual, pred)

    classif_report = classification_report(actual, pred)
    conf_matrix = confusion_matrix(actual, pred)

    return acc_score, classif_report, conf_matrix

# ==================== Get Model ====================
def get_model(config):
    yaml_data= read_yaml(config)

    active_model = yaml_data['base']['active_model']
    return active_model

def split_data(config):
    yaml_data = read_yaml(config)
    train_data_path = yaml_data['split_data']['train_data_path']
    test_data_path = yaml_data['split_data']['test_data_path']

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    x_train = train_df.drop(columns = 'results')
    x_test = test_df.drop(columns = 'results')
    y_train = train_df['results']    # Returns series
    y_test = test_df['results']    # Returns series

    # print(x_train.shape)
    # print(x_test.shape)
    # print("="*50)

    # print(x_train.columns)
    # print("="*50)


    return x_train, x_test, y_train, y_test

# ==================== Train and evaluate model ====================
def train_and_evaluate(config):
    yaml_data = read_yaml(config)

    active_model = get_model(config)
    random_state = 1

    # Get x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = split_data(config = config)

    

    # Yaml will always return string
    # so active_model = str and hence active_model(), will give error "String object is not callable"
    # So we will map
    model_map = {
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier
    }

    # Fetch model's parameters from yaml_data
    parameters = yaml_data['models'][active_model].get('parameters', {})
    
    # Adding random_state and n_jobs in parameters
    parameters['random_state'] = random_state

    # Getting the model class from model_map
    model_class = model_map.get(active_model)
    
    if model_class:
        model = model_class(**parameters)
    
    # print(model)
    
    #Fitting the model
    model.fit(x_train, y_train)
    print(f"""{model}
Model trained successfully
          """)

    y_pred = model.predict_proba(x_test)
    print("Prediction on test data completed.")
    print("="*50)

    # Evaluating the model
    acc_score, classif_report, conf_matrix = evaluate_model(y_test, y_pred)

    print(f"""=============== Accuracy Score ===============
{acc_score}

=============== Classification Report ===============
{classif_report}

=============== Confusion Matrix ===============
{conf_matrix}
==================================================""")

    # print(f"Accuracy Score: {acc_score}")
    # print('*'*25)
    # print(f"Classification Report:{classif_report}")
    # print('*'*25)
    # print(f"Confusion matrix: {conf_matrix}")
    # print('*'*25)
    # print(model.get_params())

    # print(pd.DataFrame(conf_matrix))

    return model, acc_score, classif_report, conf_matrix

# ==================== Save reports ====================
def save_reports(model, acc_score, classif_report, conf_matrix, config):
    yaml_data = read_yaml(config)

    active_model = yaml_data['base']['active_model']
    parameters = model.get_params()
    model_path = yaml_data['model_path']
    
    # Saving the model
    os.makedirs(model_path, exist_ok = True)
    model_file_path = os.path.join(model_path, "model.joblib")
    joblib.dump(model, model_file_path)

    print(f"Model save successfully: {model_file_path}")
    print("="*50)

    # Saving model_name, accuracy, classification_report in a same file
    model_scores_path = yaml_data['reports']['model_scores_path']
    parameters = model.get_params()
    os.makedirs('reports', exist_ok = True)

    metrics = {
        "model_name": active_model,
        "accuracy": acc_score,
        "classification_report": classif_report
        }
    
    with open(model_scores_path, 'w') as f:
        json.dump(metrics, f, indent = 4)
    
    print(f'Scores saved successfully: {model_scores_path}')
    print("="*50)

    # Saving models parameters in parameters.json
    model_parameters_path = yaml_data['reports']['model_parameters_path']
    params = {
        "model_name": active_model,
        'model_parameters': parameters
    }

    with open(model_parameters_path, 'w') as f:
        json.dump(params, f, indent = 4)
    
    print(f'Model parameters saved successfully: {model_parameters_path}')
    print("="*50)


    # Saving confusion matrox saperately in a csv file
    conf_matrix_path = yaml_data['reports']['conf_matrix_path']
    
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index = ["Actual_0", "Actual_1"],
        columns = ["Predicted_0", "Predicted_1"]
        )
    
    conf_matrix_df.to_csv(conf_matrix_path, index = True, header= True)

    print(f"Confusion matrix saved successfully: {conf_matrix_path}")
    print("="*50)

# ==================== Main function ====================
def main(config):
    model, acc_score, classif_report, conf_matrix = train_and_evaluate(config)
    save_reports(model, acc_score, classif_report, conf_matrix, config)

    print("""=============== Final Output ===============

1) Model Trained,
2) Parameters saved
3) Accuracy Score, Classification_report and Confusion_matrix saved."""
          )
    

# ==================== Calling the main function ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = 'params.yaml')

    args = parser.parse_args()

    # Calling main functon
    main(config = args.config)