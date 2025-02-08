from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
from autocorrect import Speller
import unidecode
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

app = Flask(__name__)
app = Flask(__name__, static_url_path='/assets')


def preprocess_data(data):
    # Preprocessing functions
    def case_convert():
        data.tweet_text = [i.lower() for i in data.tweet_text.values]

    def remove_specials():
        data.tweet_text = [re.sub(r"[^a-zA-Z]", " ", text) for text in data.tweet_text.values]

    def remove_shorthands():
        CONTRACTION_MAP = {
            # contraction mapping
        }
        texts = []
        for text in data.tweet_text.values:
            string = ""
            for word in text.split(" "):
                if word.strip() in list(CONTRACTION_MAP.keys()):
                    string = string + " " + CONTRACTION_MAP[word]
                else:
                    string = string + " " + word
            texts.append(string.strip())
        data.tweet_text = texts

    def remove_stopwords():
        texts = []
        stopwords_list = stopwords.words('english')
        for item in data.tweet_text.values:
            string = ""
            for word in item.split(" "):
                if word.strip() in stopwords_list:
                    continue
                else:
                    string = string + " " + word
            texts.append(string)
        data.tweet_text = texts

    def remove_links():
        texts = []
        for text in data.tweet_text.values:
            remove_https = re.sub(r'http\S+', '', text)
            remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
            texts.append(remove_com)
        data.tweet_text = texts

    def remove_accents():
        data.tweet_text = [unidecode.unidecode(text) for text in data.tweet_text.values]

    def normalize_spaces():
        data.tweet_text = [re.sub(r"\s+", " ", text) for text in data.tweet_text.values]

    # Apply preprocessing steps
    case_convert()
    remove_links()
    remove_shorthands()
    remove_accents()
    remove_specials()
    remove_stopwords()
    normalize_spaces()


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    loaded_model = joblib.load("F:\cyber\cyberbullying_model.pkl")
    vec = joblib.load("vectorizer.pkl")
    lenc = joblib.load("label_encoder.pkl")
    preprocessed_text = vec.transform([text]).toarray()
    predicted_label = loaded_model.predict(preprocessed_text)
    predicted_category = lenc.inverse_transform(predicted_label)[0]
    return render_template('result.html', text=text, category=predicted_category)



if __name__ == '__main__':
    app.run(debug=True)
