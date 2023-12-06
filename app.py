from flask import Flask, render_template, request
import joblib
import numpy as np
import nltk
import string

import numpy as np 
import pandas as pd 

import os
from pathlib import Path

import string
import nltk
from nltk.corpus import stopwords

import scipy.io
import scipy.linalg
from scipy.sparse import csr_matrix, vstack, lil_matrix 
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

import plotly.express as px
import plotly.figure_factory as ff
from yellowbrick.text import TSNEVisualizer

app = Flask(__name__, template_folder='web')


@app.route('/')
def student():
    return render_template("home.html")

def processText(text):
    # stemming
    clean_text = clean_tweet(text)

    # vectorization
    vectorizer = joblib.load('model_vectorizer.sav')
    vector_idf = vectorizer.transform([clean_text])

    return vector_idf

def ValuePredictor(to_predict_list):
    processed_text = processText(to_predict_list[0])
    loaded_model = joblib.load('finalized_model.sav')
    result = loaded_model.predict(processed_text)
    print("Prediction : ", result[0])
    return result[0]




# Remove stop words, special chars 
# stem the word tokens
# re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
def clean_tweet(sent):
    stemmer = nltk.PorterStemmer()        
    tknzr = nltk.RegexpTokenizer(r'[a-zA-Z0-9]+')

    exclp = list(string.punctuation)     
    exclc = [
        "'re", "n't", "'m", "'s", "n't", "'s", 
        "``", "''", "'ve", "'m", "'ll", "'ve", 
        "...", "http", "https"]    
    sw = set(stopwords.words("english") + exclp + exclc)    

    tokens = tknzr.tokenize(sent.lower())
    words = [stemmer.stem(token) for token in tokens if not token in sw]
    return " ".join(words)


@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        result = round(float(ValuePredictor(to_predict_list)), 2)
        if result == 0:
            result = "Niether, All Good Bhai :)"
        if result == 1:
            result = "Offensive"
        if result == 2:
            result = "Hateful"
        return render_template("home.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)