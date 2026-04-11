import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(words)

def predict_sentiment(text):
    processed = preprocess_text(text)
    vectorized = tfidf.transform([processed])
    prediction = model.predict(vectorized)[0]
    
    return prediction

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("🎬 Sentiment Analysis App")

user_input = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        result = predict_sentiment(user_input)
        
        if result == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😡")
    else:
        st.warning("Please enter some text")