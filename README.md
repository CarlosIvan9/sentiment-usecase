# Movie Reviews Evaluator

This project performs binary classification to determine if a movie review is positive or negative. 

---

## Table of Contents
- [Project Overview](#project-overview)
- [Setup & Installation](#setup--installation)
- [Project Structure](#project-structure)
- [Models & Benchmarking](#models--benchmarking)
- [Demo](#demo)
- [Version Control Workflow](#version-control-workflow)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

---

## Project Overview

This project performs binary classification to determine if a movie review is positive or negative. Several pre-trained NLP models were compared on a test dataset consisting of 100 movie reviews. After that, the selected model was deployed as a REST API.

---

## Setup & Installation

### Python Version
This project was developed and tested with Python 3.11

### Virtual environments
There are 2 different requirements files: 

* requirements.txt : libraries needed for the deployment of the api
* requirements-dev.txt : libraries needed to run development scripts

---

## Project structure

All that is needed to run any of the scripts can be found in the repository, except for the tokens to access Hugging Face and Cohere models. We decided to keep also the data here since the size of the files is very small (input data is only 130KB).

The structure of the repository is explained as follows:

* **data:** includes both input data and output data (models' predictions, metadata)
* **docs:** a presentation of the project
* **mlruns:** files needed to visualize the mlflow dashboard
* **scripts:** scripts used to generate predictions for each model, calculate metrics, and log them to mlflow

The deployed app and requirements are on the root location.

---

## Approach

We evaluated five different sentiment analysis models. Four of these consisted of transformer-based models hosted on Hugging Face, plus a generative model of Cohere.

Predictions were generated and evaluated on a test set of 100 reviews from the IMDB movie reviews dataset, each labeled with a ground-truth sentiment.

---

## Metrics

To ensure a robust comparison, we tracked and compared each model using the following performance metrics:

Classification metrics:

Precision, Recall, and F1-Score (both per class and aggregated)

Regression-style metric:

Mean Absolute Error (MAE) on the raw prediction scores (both per class and overall), used to better understand the distribution and confidence levels of model outputs. MAE was selected for its interpretability and unit-consistency.

Efficiency metric:

Inference time: Measured the average time required by each model to generate predictions on the test set, critical for understanding deployment feasibility.



---

## Experiment tracking

We used MLflow for:

* Comparing metrics across different models.

* Storing configuration details and evaluation results.

MLflow’s interface allowed for transparent and reproducible experimentation, making model comparison straightforward and auditable.

---

## Model selection

From the five models evaluated, the final model was selected based on a combination of:

* Highest weighted F1-score, ensuring balanced performance across both classes.

* Fast inference time, making it suitable for production deployment.

---

## Deployment

The selected model was deployed as a REST API using Flask. The API includes:

 * A /predict endpoint to accept user input and return sentiment predictions.

 * An optional web application interface, allowing users to interact with the model through a simple, user-friendly UI.

This deployment ensures both programmatic and interactive access to the model.

The url of the api is the following:

https://sentiment-usecase.onrender.com/


