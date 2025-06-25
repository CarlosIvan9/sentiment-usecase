# Movie Reviews Evaluator

This project performs binary classification on movie reviews to determine whether they are positive or negative. 

---

## Table of Contents
- [Project overview](#project-overview)
- [Setup & installation](#setup--installation)
- [Project structure](#project-structure)
- [Models and approaches](#models-and-approaches)
- [Metrics](#metrics)
- [Experiment tracking](#experiment-tracking)
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

## Models and approaches

We evaluated five different approaches using several LLMs. Four of these consisted of transformer-based models hosted by Hugging Face. The last approach was a generative model hosted by Cohere.

Predictions were generated and evaluated on a test set of 100 reviews from the IMDB movie reviews dataset, each labeled with a ground-truth sentiment.

### Models

* **cardiffnlp/twitter-roberta-base-sentiment**: roBERTa-base model trained on ~58M tweets and finetuned for sentiment analysis. Outputs 3 classes (positive, neutral, negative).
* **distilbert/distilbert-base-uncased-finetuned-sst-2-english**: DistilBERT-base-uncased fine-tuned on the SST-2 dataset for sentiment analysis. Outputs 2 classes (positive, negative).
* **MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli**: foundation model DeBERTa-v3-large (improved alternative over BERT, RoBERTa) finetuned on several NLI datasets. Used for zero shot classification.
* **Command A**: generative model by Cohere.

### Approaches

#### Approach 1:
#### Approach 2:
#### Approach 3:
#### Approach 4:
#### Approach 5:

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

## Model selection

From the five approaches evaluated, the final model was selected based on a combination of:

* Highest weighted F1-score, ensuring balanced performance across both classes.

* Fast inference time, making it suitable for production deployment.

This model was DeBERTa-v3-large-mnli-fever-anli-ling-wanli using as tailored labels. This model came up second in weighted F1-score. However, the inference time of the model with best F1-score was significantly longer than this one due to rate limits. 

---

## Deployment

The selected model was deployed as a REST API using Flask. The API includes:

 * A /predict endpoint to accept user input and return sentiment predictions.

 * An optional web application interface, allowing users to interact with the model through a simple, user-friendly UI.

This deployment ensures both programmatic and interactive access to the model.

The url of the api is the following:

https://sentiment-usecase.onrender.com/

---

## Future work

asdfsdfdfdfddf
