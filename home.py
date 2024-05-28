from flask import Flask, render_template, jsonify, request
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process-text', methods=['POST'])
def process_text():
    input_text = request.form.get('input-text')
    print(input_text)

    df = pd.read_csv('final_random_spam_filtering_dataset.csv')
    df['label'] = df.label.map({'ham': 0, 'spam': 1})
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # Vectorize the text data using CountVectorizer and TfidfVectorizer
    vectorizer_count = CountVectorizer()
    vectorizer_tfidf = TfidfVectorizer()
    X_train_count = vectorizer_count.fit_transform(X_train)
    X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)

    # Load the models for future use
    loaded_knn_count_model = joblib.load('models/knn_count_model.joblib')
    loaded_lr_tfidf_model = joblib.load('models/lr_tfidf_model.joblib')
    loaded_nb_count_model = joblib.load('models/nb_count_model.joblib')
    loaded_knn_tfidf_model = joblib.load('models/knn_tfidf_model.joblib')
    loaded_lr_count_model = joblib.load('models/lr_count_model.joblib')
    loaded_nb_tfidf_model = joblib.load('models/nb_tfidf_model.joblib')

    new_text = [input_text]
    new_text_count = vectorizer_count.transform(new_text)
    new_text_tfidf = vectorizer_tfidf.transform(new_text)

    # Use the trained models to predict whether the new text is spam or ham
    lr_count_prediction = loaded_lr_count_model.predict(new_text_count)
    lr_tfidf_prediction = loaded_lr_tfidf_model.predict(new_text_tfidf)
    nb_count_prediction = loaded_nb_count_model.predict(new_text_count)
    nb_tfidf_prediction = loaded_nb_tfidf_model.predict(new_text_tfidf)
    knn_count_prediction = loaded_knn_count_model.predict(new_text_count)
    knn_tfidf_prediction = loaded_knn_tfidf_model.predict(new_text_tfidf)

    lr_count_scores = loaded_lr_count_model.predict_proba(new_text_count)
    lr_tfidf_scores = loaded_lr_tfidf_model.predict_proba(new_text_tfidf)

    nb_count_scores = loaded_nb_count_model.predict_proba(new_text_count)
    nb_tfidf_scores = loaded_nb_tfidf_model.predict_proba(new_text_tfidf)

    knn_count_scores = loaded_knn_count_model.predict_proba(new_text_count)
    knn_tfidf_scores = loaded_knn_tfidf_model.predict_proba(new_text_tfidf)

    zeros_prob = []
    ones_prob = []
    lr_count_zero, lr_count_one = np.split(lr_count_scores[0], 2)
    lr_tfidf_zero, lr_tfidf_one = np.split(lr_tfidf_scores[0], 2)
    nb_count_zero, nb_count_one = np.split(nb_count_scores[0], 2)
    nb_tfidf_zero, nb_tfidf_one = np.split(nb_tfidf_scores[0], 2)
    knn_count_zero, knn_count_one = np.split(knn_count_scores[0], 2)
    knn_tfidf_zero, knn_tfidf_one = np.split(knn_tfidf_scores[0], 2)
    zeros_prob.extend([lr_count_zero, lr_tfidf_zero, nb_count_zero, nb_tfidf_zero, knn_count_zero, knn_tfidf_zero])
    ones_prob.extend([lr_count_one, lr_tfidf_one, nb_count_one, nb_tfidf_one, knn_count_one, knn_tfidf_one])
    zeros_prob = [round(score.tolist()[0], 2) for score in zeros_prob]
    ones_prob = [round(label.tolist()[0], 2) for label in ones_prob]
    accumulate_result = []
    accumulate_result.extend(
        [lr_count_prediction, lr_tfidf_prediction, nb_count_prediction, nb_tfidf_prediction, knn_count_prediction,
         knn_tfidf_prediction])
    accumulate_result = [round(label.tolist()[0], 2) for label in accumulate_result]
    final_result = 'not spam' if accumulate_result.count(0) > accumulate_result.count(1) else 'spam'
    print(zeros_prob, ones_prob, accumulate_result, final_result)

    # create a response dictionary with the scores
    response = {
        'zeros_prob': zeros_prob,
        'ones_prob': ones_prob,
        'final_result': final_result,
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)