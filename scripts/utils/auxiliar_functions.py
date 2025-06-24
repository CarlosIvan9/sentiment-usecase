

import pandas as pd
import os
import time
import json
from pathlib import Path



def load_test_data():
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