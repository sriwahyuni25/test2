import pandas as pd
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

# Import data
positif = pd.read_csv('pemilu2024_negative.csv')
negatif = pd.read_csv('pemilu2024_positive.csv')

# Merge data
data = pd.concat([positif, negatif])

# Drop duplicate data
data_final = data[['sentiment', 'steming_data']].dropna()
data_final.sentiment.value_counts()

data_final['sentiment'] = data_final.sentiment.map({'positif': 1, 'negatif': 0})

x = data_final.steming_data
y = data_final['sentiment']

# Vectorizing
vec = CountVectorizer().fit(x)
x_features = vec.get_feature_names_out()
x_vec = vec.transform(x)
tfidf = TfidfTransformer().fit(x_vec)
tfidf_data = tfidf.transform(x_vec)

x_train, x_test, y_train, y_test = train_test_split(x_vec, y, test_size=0.2, random_state=1)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_result = model.predict(x_test)

accuracy_rf = accuracy_score(y_test, y_result)

# Menghitung akurasi pelatihan
train_accuracy_rf = model.score(x_train, y_train)

# Memprediksi label untuk data uji
y_pred = model.predict(x_test)

validation_accuracy_rf = accuracy_score(y_test, y_pred)

# Simpan model ke file
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Simpan vectorizer
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vec, vec_file)

# Simpan tfidf transformer
with open('tfidf_transformer.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

# Load model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load vectorizer and tfidf transformer
with open('vectorizer.pkl', 'rb') as vec_file:
    vec = pickle.load(vec_file)

with open('tfidf_transformer.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

# Membuat objek stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    # Menghilangkan karakter non-alphabet
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenisasi
    words = word_tokenize(text)
    # Stemming
    words = [stemmer.stem(word) for word in words]
    # Menggabungkan kembali menjadi satu string
    return ' '.join(words)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def cek():
    return jsonify("sukses")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']

    # Preprocessing teks
    preprocessed_text = preprocess_text(text)
    
    # Transform input text
    x_vec = vec.transform([preprocessed_text])
    tfidf_data = tfidf.transform(x_vec)

    # Predict sentiment
    prediction = model.predict(tfidf_data)
    sentiment = 'positif' if prediction[0] == 1 else 'negatif'

    return jsonify({'sentiment': sentiment})

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    # Check if a file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        # Read the CSV file
        data = pd.read_csv(file)

        # Check if 'text' column exists in the CSV
        if 'text' not in data.columns:
            return jsonify({'error': 'CSV file must contain a "text" column'}), 400

        # Transform and predict each text entry in the CSV
        data['text'] = data['text'].apply(preprocess_text)
        x_vec = vec.transform(data['text'])
        tfidf_data = tfidf.transform(x_vec)
        predictions = model.predict(tfidf_data)

        # Map predictions to sentiment labels
        data['sentiment'] = predictions
        data['sentiment'] = data['sentiment'].map({1: 'positif', 0: 'negatif'})

        # Convert result to dictionary
        result = data.to_dict(orient='records')

        return jsonify(result)

@app.route('/wordcloud', methods=['POST'])
def generate_wordcloud():
    data = request.get_json(force=True)
    text = data['text']

    # Generate word cloud
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = None, 
                min_font_size = 10).generate(text)

    # Plot word cloud
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    # Save word cloud image
    wordcloud.to_file('static/wordcloud.png')

    return jsonify({'message': 'Word cloud generated successfully'})


if __name__ == '__main__':
    app.run(debug=True)