from flask import Flask, request, jsonify

from huggingface_hub import InferenceClient

my_token='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'

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, Foo!'

def get_sentiment(review_raw):
    # Make sure input is a list (output of hf is 3 classes if a string is given, or just the top class if a list is given)
    if not isinstance(review_raw, list):
        review_raw = [review_raw]
    client=InferenceClient(
        model='cardiffnlp/twitter-roberta-base-sentiment',
        token=my_token
        )
    output_hf=client.text_classification(review_raw)
    sentiments=[]
    for a_review in output_hf:
        if a_review.label == 'LABEL_2':
            sentiments.append('Positive')
        elif a_review.label == 'LABEL_1':
            sentiments.append('Neutral')
        elif a_review.label == 'LABEL_0':
            sentiments.append('Negative')
    return sentiments



@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()
    review = body.get("review", "")

    if not review:
        return jsonify({"error": "Missing review"}), 400
    
    sentiments = get_sentiment(review)

    return jsonify(sentiments), 200

if __name__ == "__main__":
    app.run(debug=True)



# To run app you have to do this in the termal (powershell)
#1- $env:FLASK_APP = "my_app.py"
#2- flask run

# Alternatively you can run your app directly with Python (especially useful for larger apps):
# python my_app.py
# But for that to work, you need this at the bottom of my_app.py:
# if __name__ == "__main__":
#    app.run(debug=True)


# To test a post (in terminal. Git bash works well, powershell does not):
#curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"review": "I am alright and you?"}'
# In a list, double members inside it should appear in double quotes
#curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"review": ["Me again", "Trump is blonde", "I hated it"]}'
