"""
auxiliar_functions.py

Package that contains functions that are used every time predictions of a new model are calculated. 

"""

import pandas as pd
import os
import time
import json
from pathlib import Path



def load_test_data():
    """
    Loads test data into a pandas dataframe. 

    Returns:
        df (pandas dataframe): dataframe test data.
    """
    try:
        # Works in regular Python scripts
        base_dir = Path(__file__).resolve().parent.parent.parent
    except NameError:
        # Fallback for Jupyter notebooks and interactive shells
        base_dir = Path().resolve().parent.parent

    # Now use it to build your file path
    data_path = base_dir / "data" / 'inputs' / "IMDB-movie-reviews.csv"

    # Read file
    data = pd.read_csv(data_path, sep=';', encoding='latin-1')


    data.rename(columns={'sentiment':'Target'}, inplace=True)

    data['review_index'] = data.index
    return data 



def save_outputs(data, predictions, model_name, adaptations, inference_time, other_comments):
    """
    This function is used after calculating predictions for the test data. It saves the predictions as a pandas df. It 
    also saves other information such as model used, time it took to process the test data, and additional notes, into
    a json. These outputs are then used to calculate metrics that are logged into mlflow.

    Args:
        data (pandas df): dataframe with reviews and ground truths
        predictions (pandas df): dataframe with reviews and predictions
        model_name (str): Used as a folder that will contain the outputs. It is usually the name of the model. 
        adaptations (str): if there where specific adaptations of the model or prompt to generate predictions.
        inference_time (float): number of seconds it took to calculate predictions to all reviews in the test data
        other_comments (str): other comments regarding the model or its results

    """
    # Make output dir
    try:
        # Works in regular Python scripts
        base_dir = Path(__file__).resolve().parent.parent.parent
    except NameError:
        # Fallback for Jupyter notebooks and interactive shells
        base_dir = Path().resolve().parent.parent
    # Now use it to build file path to store predictions and metadata
    path_outputs = base_dir / "data" / 'outputs' / 'runs' / model_name
    # Create the directory (and parents if they don't exist)
    path_outputs.mkdir(parents=True, exist_ok=True)

    # Add target to the predictions df
    output_df=data.merge(predictions.drop(columns=['review']), how='left' , on='review_index')
    output_df.drop(columns=['review_index'], inplace=True)

    # Save predictions
    output_df.to_csv(path_outputs / 'predictions.csv', index=False, sep=';')

    # Create metadata file and save it
    metadata = { 
        'model':model_name,
        'adaptations': adaptations,
        'inference_time': inference_time,
        'other_comments': other_comments
    }
    # Save to JSON file
    with open(path_outputs / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4) 