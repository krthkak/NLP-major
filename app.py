import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json,re

app = Flask(__name__)
PS = pickle.load(open('portStem.pkl', 'rb'))
CV = pickle.load(open('Countvec.pkl','rb'))
model = pickle.load(open('Gaussclass.pkl','rb'))
with open("stopwords.json","r") as file:
    stop_words = json.load(file)["words"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    senti = {0:"Negative",1:"Positive"}
    out = request.form.get('review')
    
    review = re.sub('[^a-zA-Z]', ' ' ,out)
    #return render_template('index.html', prediction_text='The Sentiment of the review is {}'.format(out))
    review = review.split()
    review = [PS.stem(word) for word in review if not word in set(stop_words)]
    review = " ".join(review)
    
    corpus = [[review]]
    x=CV.fit_transform(corpus).toarray()
    pred = model.predict(x)

    output = senti[pred[0]]

    return render_template('index.html', prediction_text='The Sentiment of the review is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

