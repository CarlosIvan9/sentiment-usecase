# Movie Reviews Evaluator

This project performs binary classification to determine if a movie review is positive or negative. 

---

## Table of Contents
- [Project Overview](#project-overview)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Models & Benchmarking](#models--benchmarking)
- [Project Structure](#project-structure)
- [Demo](#demo)
- [Version Control Workflow](#version-control-workflow)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

---

## Project Overview

This project performs binary classification to determine if a movie review is positive or negative. For this, several pre-trained NLP models were tested. Model selection was done on a test dataset consisting of 100 movie reviews.

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


---

## ⚙️ Setup & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/emotion-evaluator.git
cd emotion-evaluator
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
