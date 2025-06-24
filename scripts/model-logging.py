"""
model-logging.py

Given a folder containing predictions and metadata of a specific model, it generates metrics and uploads them to 
mlflow for benchmarking.

Usage:
    python model-logging.py --model_name_folder
"""
import mlflow
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import argparse



def load_predictions_and_metadata(model_name):
    """
    Loads predictions and metadata needed for a particular model. 

    Args:
        model_name (str): folder containing predictions and metadata of a specific model

    Returns:
        predictions (pandas dataframe): dataframe containing reviews, their sentiment, probability of being positive, 
        and predicted sentiment.
        metadata (dictionary): metadata and other information such as inference time
    """
    # Make output dir
    try:
        # Works in regular Python scripts
        base_dir = Path(__file__).resolve().parent.parent
    except NameError:
        # Fallback for Jupyter notebooks and interactive shells
        base_dir = Path().resolve().parent

    path_predictions = base_dir / "data" / 'outputs' / 'runs' / model_name

    predictions = pd.read_csv(path_predictions / 'predictions.csv' , sep=';')
    with open(path_predictions / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return predictions, metadata


def flatten_dict(d, parent_key='', sep='-'):
    """
    Flattens a dictionary.  Used in log_model_to_mlflow to log metrics into mlflow
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_l1(df):
    """
    Calculates the mean absolute error for a particular dataframe with predictions. 

    Args:
        df (pandas df): dataframe with predictions

    Returns:
        l1_loss (float): mean absolute error 
    """
    # Convert string labels to numeric: 1 if 'positive', 0 if 'negative'
    y_true = (df["Target"] == "positive").astype(int)
    y_pred = df["positive_score"]

    # Calculate L1 loss
    l1_loss = np.abs(y_true - y_pred).mean()
    return l1_loss

def l1_per_label(df):
    """
    Calculates the mean absolute error for a particular dataframe with predictions, slicing per class. 

    Args:
        df (pandas df): dataframe with predictions

    Returns:
        mae_positive (float): mean absolute error of records with positive review
        mae_negative (float): mean absolute error of records with negative review
    """
    # Convert string labels to numeric
    y_pred = df["positive_score"]

    # Create a mask for each class
    positive_mask = df["Target"] == "positive"
    negative_mask = df["Target"] == "negative"

    # MAE for positive samples (where true label = 1)
    mae_positive = np.abs(y_pred[positive_mask] - 1).mean()

    # MAE for negative samples (where true label = 0)
    mae_negative = np.abs(y_pred[negative_mask] - 0).mean()

    return mae_positive, mae_negative


def log_model_to_mlflow(predictions, metadata):
    """
    Logs metrics and metadata of a model into mlflow. 

    Args:
        predictions (pandas df): dataframe with predictions
        metadata (dict): dictionary with metadata and other interesting information to be logged.

    """
    class_metrics=classification_report(predictions.Target, predictions.Prediction, output_dict=True)
    flat_class_metrics =  flatten_dict(class_metrics)

    # Create scores distribution plot
    sns.histplot(data=predictions, x="positive_score", hue="Target", common_norm=False, bins=15)
    plt.title("KDE by Category")
    plt.xlabel("Value")
    plt.ylabel("Density")
    # Save plot to in-memory bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)  # rewind buffer to the beginning
    plt.close()  # close the plot to free memory
    # Convert BytesIO to PIL Image
    img = Image.open(buf)

    with mlflow.start_run(run_name=metadata['model']+'/'+metadata['adaptations']):
        # Log metrics
        for akey in flat_class_metrics.keys():
            mlflow.log_metric(akey, flat_class_metrics[akey])

        mlflow.log_metric("inference_time", metadata['inference_time'])
        mlflow.log_metric("MAE", get_l1(predictions))

        mae_positive, mae_negative  = l1_per_label(predictions)
        mlflow.log_metric("MAE_positive", mae_positive)
        mlflow.log_metric("MAE_negative", mae_negative)

        # Log plots
        mlflow.log_image(img, "scores_distribution.png")

        # Log metadata
        mlflow.set_tag("model", metadata['model'])
        mlflow.set_tag("adaptations", metadata['adaptations'])
        mlflow.set_tag("other_comments", metadata['other_comments'])


def parse_args():
    parser = argparse.ArgumentParser(description="Run sentiment model logging")
    parser.add_argument('--model_name_folder', type=str, help='Name of the model', required=True,)
    return parser.parse_args()


#model_name_folder = 'twitter-roberta'
experiment_name = "sentiment-usecase"



if __name__ == "__main__":
    args = parse_args()

    predictions, metadata = load_predictions_and_metadata(args.model_name_folder)

    mlflow.set_experiment(experiment_name)

    log_model_to_mlflow(predictions, metadata)   







