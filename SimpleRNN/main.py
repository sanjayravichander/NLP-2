import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model

# Load the model and encoders
model = load_model('SimpleRNN_imdb.h5',compile=False)

# Maping of word index back to words for understanding
word_index = imdb.get_word_index()

reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user unit
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(review):
    preprocess_input=preprocess_text(review)
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]

import streamlit as st
st.title("Sentiment Analysis")
st.write("Enter a movie review to predict whether it is positive or negative")
user_input=st.text_area("Enter your review here:")
if st.button("Predict"):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {prediction[0][0]:.2f}")
else:
    st.write("Please enter a review and click predict")