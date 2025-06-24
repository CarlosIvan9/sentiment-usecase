
"""
predict-distilbert-sst.py

Predicts review sentiments on the test data using the sentiment analysis model
    distilbert/distilbert-base-uncased-finetuned-sst-2-english . Returns predictions in a pandas 
    df with a format ready to benchmark, and other information of relevance to be analyzed such as inference time.

Usage:
    python predict-distilbert-sst.py 
"""


from huggingface_hub import InferenceClient
import pandas as pd
import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Loads from .env


# Add scripts/utils/ to path, to load package functions
import sys
sys.path.append(str(Path(__file__).resolve().parent / "utils"))
from auxiliar_functions import load_test_data, save_outputs

my_token=os.getenv("HUGGING_FACE_TOKEN")



def get_sentiment(reviews):
    """
    Calculates sentiment of a review or a list of reviews using the sentiment model
    distilbert/distilbert-base-uncased-finetuned-sst-2-english . Length of the reviews is truncated to 2k characters
    to avoid issues with long reviews. 

    Args:
        reviews (str or list): a single review (str) or a list of reviews (list)

    Returns:
        df (pandas dataframe): dataframe containing reviews, their sentiment, probability of being positive, 
        and predicted sentiment.
    """
    # Make sure input is a list (output of hf is 3 classes if a string is given, or just the top class if a list is given)
    if not isinstance(reviews, list):
        reviews = [reviews]
    client=InferenceClient(
        model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
        token=my_token
        )

    # Inference for all reviews
    all_rows = []
    for i, review in enumerate(reviews):
        review = review[:2000] # Quick and dirty truncation of long reviews (max tokens is around 500 (we assumed 3 letters per token))
        output = client.text_classification(review, top_k=None)
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
    df = df[df["label"] == "POSITIVE"].reset_index(drop=True)

    # Format output df 
    df.drop(columns=['label'], inplace=True)
    df.rename(columns={'score':'positive_score'}, inplace=True)

    outputs_list = ['positive' if score > 0.5 else 'negative' for score in df['positive_score']]
    df['Prediction']=outputs_list
    return df




model_name = 'distilbert-finetuned-sst-2'
adaptations = 'Truncation to 2k characters'
other_comments = 'Model assigns negative sentiment to reviews with connotative negative words (like violence, drugs...) regardless if the review is positive. But it seems better than roberta'




if __name__ == "__main__":
    data = load_test_data()

    start = time.time()
    predictions = get_sentiment(list(data.review))
    end = time.time()
    inference_time = end - start

    save_outputs(data, predictions, model_name, adaptations, inference_time, other_comments)









