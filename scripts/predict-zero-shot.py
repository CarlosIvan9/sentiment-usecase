
"""
predict-zero-shot.py

Predicts review sentiments on the test data using the zero shot classificator model
    MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli. Returns predictions in a pandas df with a format ready to 
     benchmark, and other information of relevance.

Usage:
    python predict-zero-shot.py 
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






def get_sentiment(reviews, positive_label, negative_label):
    """
    Calculates sentiment of a review or a list of reviews using the zero shot classificator model
    MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

    Args:
        reviews (str or list): a single review (str) or a list of reviews (list)
        positive_label (str): label indicating positive review
        negative_label (str): label indicating negative review

    Returns:
        df (pandas dataframe): dataframe containing reviews, their sentiment, probability of being positive, 
        and predicted sentiment.
    """

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



