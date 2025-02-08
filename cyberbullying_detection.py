from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import re
import unidecode
import numpy as np
import matplotlib.pyplot as plt
import joblib

app = Flask(__name__)
app = Flask(__name__, static_url_path='/assets')

# Load the data
data = pd.read_csv("cyberbullying_tweetswithlaw.csv")

# Preprocessing functions
def preprocess_data(data):
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

preprocess_data(data)


# Encode the target labels
lenc = LabelEncoder()
data.cyberbullying_type = lenc.fit_transform(data.cyberbullying_type)

# Visualize class balance
plt.figure(figsize=(15, 5))
un, count = np.unique(data.cyberbullying_type.values, return_counts=True)
plt.bar([lenc.classes_[int(i)] for i in un], count)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Balance")
plt.show()

# Generate word cloud for each class
for c in range(len(lenc.classes_)):
    string = ""
    for i in data[data.cyberbullying_type == c].tweet_text.values:
        string = string + " " + i.strip()

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=set(stopwords.words('english')),
                          min_font_size=10).generate(string)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(lenc.classes_[c])
    plt.show()
    del string

# Vectorize text data
vec = TfidfVectorizer(max_features=3000)
X_train, X_test, Y_train, Y_test = train_test_split(data.tweet_text.values,
                                                    data.cyberbullying_type.values,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42)

X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()

# Train the model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, Y_train)

# Print evaluation metrics
print("Train Accuracy  : {:.2f} %".format(accuracy_score(model.predict(X_train), Y_train) * 100))
print("Test Accuracy   : {:.2f} %".format(accuracy_score(model.predict(X_test), Y_test) * 100))
print("Precision       : {:.2f} %".format(precision_score(model.predict(X_test), Y_test, average='macro') * 100))
print("Recall          : {:.2f} %".format(recall_score(model.predict(X_test), Y_test, average='macro') * 100))

# Save the model
joblib.dump(model, "cyberbullying_model.pkl")
joblib.dump(vec, "vectorizer.pkl")
joblib.dump(lenc, "label_encoder.pkl")

# Function to classify text based on the trained model
def classify_text(text):
    preprocessed_text = vec.transform([text]).toarray()
    predicted_label = model.predict(preprocessed_text)
    return lenc.inverse_transform(predicted_label)[0]

# Route for home page
@app.route('/')
def home():
   return render_template('index.html')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/about')
def about_page():
    return render_template('about.html')



# Route for classification
@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    predicted_category = classify_text(text)
    
    if predicted_category in ["other_cyberbullying", "ethnicity", "gender", "age", "religion"]:
     # Render a page with JavaScript to display a dialog box before redirecting
        return render_template('dialogue_box.html', text=text, category=predicted_category)
    
    # If not cyberbullying, directly redirect to the result page
    return render_template('result.html', text=text, category=predicted_category)


if __name__ == '__main__':
    app.run(debug=True)