from flask import Flask, request, render_template_string, jsonify

from huggingface_hub import InferenceClient

my_token='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'




app = Flask(__name__)

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


# HTML form template
form_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment App</title>
</head>
<body>
    <h2>Sentiment App</h2>
    <h3>Enter a review</h3>
    <form method="POST" action="/predict">
        <input type="text" name="review" required>
        <button type="submit">Sentiment</button>
    </form>
    {% if sentiment %}
        <h4> {{ review }}: {{ sentiment }}</h4>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    return render_template_string(form_html, sentiment=None)

@app.route('/predict', methods=['POST'])
def predict_sentiment(): 
    if request.is_json:  # Applies for curl requests
        body = request.get_json()
        review = body.get("review", "")
        if not review:
            return jsonify({"error": "Missing review"}), 400
        sentiments = get_sentiment(review)
        return jsonify(sentiments), 200

    else: #Applies for request from html template
        review = request.form.get('review', '')  
        sentiments = get_sentiment(review)
        if len(sentiments)==1:  #Cleaner output in case a single review is given
            sentiments=sentiments[0]
        return render_template_string(form_html, sentiment=sentiments, review=review)
    


if __name__ == '__main__':
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
