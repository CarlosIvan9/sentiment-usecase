
from huggingface_hub import InferenceClient
import pandas as pd
import os
import time
import json
from pathlib import Path


my_token='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'


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



def get_sentiment(reviews, positive_label, negative_label):

    # Make sure input is a list (output of hf is 3 classes if a string is given, or just the top class if a list is given)
    if not isinstance(reviews, list):
        reviews = [reviews]
    client=InferenceClient(
        model='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
        token=my_token
        )
    
    # Inference for all reviews
    all_rows = []
    for i, review in enumerate(reviews):
        review = review[:4010] # truncation of long reviews to the maximum we have seen
        output = client.zero_shot_classification(review, candidate_labels = [positive_label, negative_label])
        for item in output:
            all_rows.append({
                "review_index": i,
                "review" : review, 
                "label": item["label"],
                "score": item["score"]
            })

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Extract positive score only
    df = df[df["label"] == positive_label].reset_index(drop=True)

    # Format output df 
    df.drop(columns=['label'], inplace=True)
    df.rename(columns={'score':'positive_score'}, inplace=True)

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



model_name = 'zero-shot'
adaptations = 'simple labels'
other_comments = 'Model seems to overcome issues of sentiment classification methods. Labels: a very positive movie review, a very negative movie review'

positive_label='a very positive movie review'
negative_label='a very negative movie review'
#positive_label='a positive movie review (regardless if the movie context is negative)'
#negative_label='a negative movie review (regardless if the movie context is positive)'




if __name__ == "__main__":
    data = load_test_data()

    start = time.time()
    predictions = get_sentiment(list(data.review), positive_label, negative_label)
    end = time.time()
    inference_time = end - start

    save_outputs(data, predictions, model_name, adaptations, inference_time, other_comments)



