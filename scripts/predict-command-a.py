

import pandas as pd
import os
import time
import json
from pathlib import Path
import cohere
from dotenv import load_dotenv
load_dotenv()  # Loads from .env

# Add scripts/utils/ to path, to load package functions
import sys
sys.path.append(str(Path(__file__).resolve().parent / "utils"))
from auxiliar_functions import load_test_data, save_outputs


cohere_api_key=os.getenv("COHERE_TOKEN")



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