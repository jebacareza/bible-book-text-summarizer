from flask import Flask, jsonify, request
import numpy as np
import pandas as pd

import os
import re
import string

#import nltk
#from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from collections import Counter

import pickle

# For tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import flask
app = Flask(__name__)

filename_pkl_best = '../output/saved_models/model_lstm_best.pkl'
clf = pickle.load(open(filename_pkl_best, 'rb'))
filename_pkl_tokenizer = '../output/saved_models/tokenizer.pkl'
loaded_tokenizer = pickle.load(open(filename_pkl_tokenizer, 'rb'))

import sys
sys.path.insert(0, '../src/modules/')
from data_preprocessing import *


###################################################
max_len = 50
# stop_words = set(stopwords.words("english"))
stop_words = ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are',
 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don',
 "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't",
 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i',
 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn',
 "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o',
 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same',
 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than',
 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't",
 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't",
 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']

def glove_vectorizer(text):
    '''
    Vectorizes a string through pre-trained GloVe embeddings
    '''    
    sequence = loaded_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")
    
    return padded

###################################################

@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    to_predict_list = request.form.to_dict()
    review_text = lowercase(to_predict_list['review_text'])
    review_text = remove_URL(review_text)
    review_text = remove_html(review_text)
    review_text = contraction_mapping(review_text)
    review_text = remove_emoji(review_text)
    review_text = remove_punct(review_text)
    review_text = remove_stopwords(review_text)
    review_text = lemmatizer(review_text)
    review_text_vectorized = glove_vectorizer(review_text)
    
    # Load the classifier
    prob = clf.predict(review_text_vectorized)
    
    # Rounds the probabilities into 0 or 1 (0 - Rotten, 1 - Fresh)
    prob_int = prob.round().astype("int")

    if prob_int[0][0] == 1:
        prediction = "Fresh"
        
    else:
        prediction = "Rotten"
       
    
    return flask.render_template('classify.html', prediction = prediction, prob = prob[0][0]*100)


if __name__ == '__main__':
    
    app.run(debug=True)
    #app.run(host='localhost', port=8081)