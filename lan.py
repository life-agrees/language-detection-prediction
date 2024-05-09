import streamlit as st
import pandas as pd
import numpy as np
import pickle
from langdetect import detect
from sklearn.linear_model import  LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


@st.cache_data
def load_model():
    with open ('model/model_class.pkl','rb') as file:
        data = pickle.load(file)

    with open ('model/label_encoder.pkl','rb')as file:
         le = pickle.load(file)

    with open ('model/tfidf_vectorizer.pkl','rb')as file:
         tfidf = pickle.load(file)

    return data,le,tfidf



data,le,tfidf = load_model()

supported_language = ['en','fr','de','es','it',
                      'pt','nl','sv','fi','pl',
                      'el','cs','ro','hu','bg']


def prediction(text):
        if isinstance(text, str) and text.strip():
            detected_language = detect(text)
            if detected_language not in supported_language:
                return "Please Enter Supported Eruopean language "
            vec = tfidf.transform([text]).toarray()
            data1 = data.predict(vec) 
            language = le.inverse_transform(data1)
            return f"The Predicted Language Of The Given Text Is:  {str(language[0])}"
        else:
            return "Please Enter A Valid Text For Prediction."




st.title("Language Detection Application")


st.markdown("""
# Welcome to the Language Detection Application! 🌍

This app can identify **several major European languages** with ease. Simply enter your text, and let the app work its magic.

Feel free to test it out and explore its capabilities. Have fun! 😊
""")




st.subheader('user text input')
user_input = st.text_input('Enter your text here', '')
submit_button = st.button('Submit')

if submit_button:
    result = prediction(user_input)
    if "please Enter Text " in result:
        st.error(result)
    else:
        st.write(result)



like = st.checkbox("do you like this app?")
button = st.button("submit")
if button:
     if like:
          st.write("Thanks. I like it too.")
     else:
          st.write("that's okay, i'll improve.")



          