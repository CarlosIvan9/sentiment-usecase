from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import ast
from huggingface_hub import InferenceClient

my_token='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'




app = Flask(__name__)


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
    return outputs_list



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

positive_label='a positive movie review (regardless if the movie context is negative)'
negative_label='a negative movie review (regardless if the movie context is positive)'


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
        sentiments = get_sentiment(review, positive_label, negative_label)
        return jsonify(sentiments), 200

    else: #Applies for request from html template
        review = request.form.get('review', '')  

        try: # In case a list was received, to be able to treat it as such
            review = ast.literal_eval(review)
        except:
            review = review

        sentiments = get_sentiment(review, positive_label, negative_label)
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


#curl -X POST https://sentiment-usecase.onrender.com/predict -H "Content-Type: application/json" -d '{"review": ["Me again", "Love", "I hated it"]}'