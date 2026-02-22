import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index=imdb.get_word_index()
reversed_word_index={value:key for key ,value in word_index.items()}
model=tf.keras.models.load_model('simple_rnn_imdb.h5')


def decode_review(encode_review):
    return ' '.join([reversed_word_index.get(i-3,'?')for i in encode_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review



# def predict_sentiment(review):
#     preprocessed_input=preprocess_text(review)
#     prediction=model.predict(preprocessed_input)
    
    
#     sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
#     return sentiment,prediction[0][0]



st.title('IMDB Movie Review Sentiment classifier')

st.write('Enter a movie review to classify it as positive or negative')

user_input=st.text_area('Movie review')


if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)
    
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    
    st.write(f"Sentiment:{sentiment}")
    st.write(f"Prediction Score:{prediction[0][0]}")
else:
    st.write('Please enter the movie review.')
    
    

