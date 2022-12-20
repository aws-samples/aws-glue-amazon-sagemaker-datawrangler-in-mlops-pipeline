import json
import os
import tarfile

import pandas as pd
import numpy as np
import xgboost
import glob

import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve

if __name__ == "__main__":
    model_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")
    print("Extracting model from path: {}".format(model_path))
    with tarfile.open(model_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=".")
    print("Loading model")
    model = joblib.load("xgboost-model")

    print("Loading test input data")
    test_path = "/opt/ml/processing/test/"
    all_files = glob.glob(test_path + "/*.csv")
    tmp = []
    for filename in all_files:
        df_tmp = pd.read_csv(filename, index_col=None, header=None)
        tmp.append(df_tmp)
    df = pd.concat(tmp, axis=0, ignore_index=True)

    y_test = df.iloc[:, 0].values
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)
    
    prediction_probabilities = model.predict(X_test)
    predictions = np.round(prediction_probabilities)

    print("Creating classification evaluation report")
    precision = precision_score(y_test, predictions, zero_division=1)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, prediction_probabilities)
    roc_auc = roc_auc_score(y_test, predictions)
    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "roc_auc": {"value": roc_auc, "standard_deviation": "NaN"},
            "confusion_matrix": {
                "0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1])},
                "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1])},
            },
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": list(fpr),
                "true_positive_rates": list(tpr),
            },
        },
    }
    

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
