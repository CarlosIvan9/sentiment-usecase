# Movie Reviews Evaluator

This project performs binary classification on movie reviews to determine whether they are positive or negative. 

---

## Table of Contents
- [Project overview](#project-overview)
- [Setup & installation](#setup--installation)
- [Project structure](#project-structure)
- [Metrics](#metrics)
- [Experiment tracking](#experiment-tracking)
- [Models and approaches](#models-and-approaches)
- [Model selection](#model-selection)
- [Deployment](#deployment)
- [Future work](#future-work)

---

## Project overview

This project performs binary classification to determine if a movie review is positive or negative. Several pre-trained NLP models were compared on a test dataset consisting of 100 movie reviews. After that, the selected model was deployed as a REST API.

---

## Setup & installation

### Python Version
This project was developed and tested with Python 3.9.5

### Virtual environments
There are 2 different requirements files: 

* requirements.txt : libraries needed for the deployment of the API
* requirements-dev.txt : libraries needed to run development scripts

---

## Project structure

All that is needed to run any of the scripts can be found in the repository, except for the tokens to access Hugging Face and Cohere models. We decided to keep also the data here since the size of the files is very small (input data is only 130KB).

The structure of the repository is explained as follows:

* **data:** includes both input data and output data (models' predictions, metadata)
* **docs:** a presentation of the project
* **mlruns:** files needed to visualize the MLflow dashboard
* **scripts:** scripts used to generate predictions for each model, calculate metrics, and log them to MLFlow

The deployed app and requirements are in the root location.

---

## Metrics

To ensure a robust comparison, we tracked and compared each model using the following performance metrics:

**Classification metrics**

Precision, Recall, and F1-Score (both per class and aggregated)

**Regression-style metric:**

Mean Absolute Error (MAE) on the raw prediction scores (both per class and overall). This gives us an idea of how far probability scores are from the ground truth and is easy to interpret.

**Efficiency metric:**

Inference time: Measured the average time required by each model to generate predictions on the test set. Critical for understanding deployment feasibility.



---

## Experiment tracking

We used MLflow for:

* Comparing metrics across different models.

* Storing configuration details and evaluation results.

MLflow’s interface allowed for transparent and reproducible experimentation, making model comparison straightforward and auditable.

---

## Models and approaches

We evaluated five different approaches using several LLMs. Four of these consisted of transformer-based models hosted by Hugging Face. The last approach was a generative model hosted by Cohere.

Predictions were generated and evaluated on a test set of 100 reviews from the IMDB movie reviews dataset, each labeled with a ground-truth sentiment.

### Models

* **cardiffnlp/twitter-roberta-base-sentiment**: roBERTa-base model trained on ~58M tweets and finetuned for sentiment analysis. Outputs 3 classes (positive, neutral, negative).
* **distilbert/distilbert-base-uncased-finetuned-sst-2-english**: DistilBERT-base-uncased fine-tuned on the SST-2 dataset for sentiment analysis. Outputs 2 classes (positive, negative).
* **MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli**: foundation model DeBERTa-v3-large (improved alternative over BERT, RoBERTa) finetuned on several NLI datasets. Used for zero shot classification.
* **Command A**: generative model by Cohere.

### Approaches

All approaches were applied on the test data sequentially, trying to improve the performance of the previous approach.

#### Approach 1: 
Applied twitter-roberta-base-sentiment. Since this approach outputs 3 classes, I had to drop the neutral class and scale the probabilities of positive and negative labels to sum to 1.
#### Approach 2: 
Applied distilbert-base-uncased-finetuned-sst-2-english, to test if a model trained specifically on 2 labels could yield a better performance. Predictive performance improved.
#### Approach 3: 
After visualizing some of the errors that were present in both of the previous approaches, we noticed that positive reviews about a movie that deals with negative topics tended to be predicted as negative. This makes sense since the purpose of both models was just to detect the sentiment of a text. Approach 3 focused on trying to correct this issue.
We applied the DeBERTa-v3-large-mnli-fever-anli-ling-wanli model. As this model can be used for zero-shot classification, we tried to overcome the previous issue by making the classes more specific and hence try to detect the sentiment of the movie review instead of the sentiment of the vocabulary used. We changed the 'positive' class for 'a very positive movie review', and did the same for the 'negative' class. The predictive performance improved.
#### Approach 4:
We tried to see if approach 3 could be improved by making even more specific classes. We changed the 'a very positive movie review' for 'a positive movie review (regardless if the movie context is negative)', and did the same for the negative class. Performance improved slightly.
#### Approach 5: 
We tested if generative models could bring an even better performance. For that, we made use of Command A from Cohere. This model gave the best performance by far. However, due to rate limits, the inference time it took for generating the outputs of the test data took significantly longer than the previous approaches.

---

## Model selection

From the five approaches evaluated, the one we will deploy was selected based on a combination of:

* Highest macro F1-score, ensuring balanced performance across both classes.

* Fast inference time, making it suitable for production deployment.

This was approach 4, the DeBERTa-v3-large-mnli-fever-anli-ling-wanli model that used tailored labels. This model came up second in macro F1-score, only behind Command A. However, the inference time of Command A was significantly longer than this one due to rate limits, and since the API is thought to receive as input both a single review and a list of reviews, we decided to sacrifice some performance for faster latency.

---

## Deployment

The selected model was deployed as a REST API using Flask and Render. The API includes:

 * A /predict endpoint to accept user input and return sentiment predictions.

 * An optional web application interface, allowing users to interact with the model through a simple, user-friendly UI.

This deployment ensures both programmatic and interactive access to the model. Furthermore, the API accepts as input either a movie review as a string, or a set of movie reviews as a list of strings.

The url of the api is the following:

https://sentiment-usecase.onrender.com/

And calls to the API can be made like this:

curl -X POST https://sentiment-usecase.onrender.com/predict -H "Content-Type: application/json" -d '{"review": ["This is the first review. Great!", "This is the second one. Awesome!", "This is the third one, and sadly, the last one..."]}'


---

## Future work

If we had more time, we would have dug deeper into the examples wrongly classified to try to detect patterns, and if possible, correct them. 
