

import pandas as pd
import os
import time
import json
from pathlib import Path
import cohere

cohere_api_key='5ou8kzfQaIX728WC1DjgQxZeNffb1bTndjFjmEu7'



def load_test_data():
    try:
        # Works in regular Python scripts
        base_dir = Path(__file__).resolve().parent
    except NameError:
        # Fallback for Jupyter notebooks and interactive shells
        base_dir = Path().resolve()

    # Now use it to build your file path
    data_path = base_dir / "data" / 'inputs' / "IMDB-movie-reviews.csv"

    # Read file
    data = pd.read_csv(data_path, sep=';', encoding='latin-1')


    data.rename(columns={'sentiment':'Target'}, inplace=True)

    data['review_index'] = data.index
    return data 


def get_sentiment(reviews):

    # Make sure input is a list (output of hf is 3 classes if a string is given, or just the top class if a list is given)
    if not isinstance(reviews, list):
        reviews = [reviews]
    co = cohere.ClientV2(cohere_api_key)
    
    prompt="""Determine if the following document is a positive or negative movie review:
    [REVIEW]

    If it is positive, return 1, and if it is negative return 0. Do not give any other answers.
    """

    # Inference for all reviews
    all_rows = []
    for i, review in enumerate(reviews):
        review = review[:8000] # truncation for maximum context length

        messages = [
            {"role": "system", "content": "You are an expert in movie reviews"},
            {"role": "user", "content": prompt.replace("[REVIEW]", review)},
        ]

        output = co.chat( model="command-a-03-2025", messages= messages  )
        output = output.message.content[0].text
        print(i)
        time.sleep(6)

        all_rows.append({
            "review_index": i,
            "review" : review, 
            "positive_score": int(output)
        })

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    outputs_list = ['positive' if score > 0.5 else 'negative' for score in df['positive_score']]
    df['Prediction']=outputs_list
    return df


def save_outputs(data, predictions, model_name, adaptations, inference_time, other_comments):

    # Make output dir
    try:
        # Works in regular Python scripts
        base_dir = Path(__file__).resolve().parent
    except NameError:
        # Fallback for Jupyter notebooks and interactive shells
        base_dir = Path().resolve()
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


model_name = 'generative-command-a'
adaptations = ''
other_comments = 'Rate limit of 10 requests per minute.'


if __name__ == "__main__":
    data = load_test_data()

    start = time.time()
    predictions = get_sentiment(list(data.review))
    end = time.time()
    inference_time = end - start

    save_outputs(data, predictions, model_name, adaptations, inference_time, other_comments)