import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # converts all to lower case
    text = nltk.word_tokenize(text)  # makes the text into list ['hi', 'hello']

    # remove special character
    new_text = []
    for i in text:
        if i.isalnum():
            new_text.append(i)

    text = new_text[:]
    new_text.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            new_text.append(i)

    text = new_text[:]
    new_text.clear()

    for i in text:
        new_text.append(ps.stem(i))

    return ' '.join(new_text)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Predict"):
    #transform
    transformed_sms = transform_text(input_sms)

    #vectorize
    vector_input = tfidf.transform([transformed_sms])

    #Predict
    result = model.predict(vector_input)[0]

    #Output
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
