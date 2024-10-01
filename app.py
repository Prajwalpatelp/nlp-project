import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Updated import

# Load the model
model = load_model('next_word_prediction_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the predict_next_word function
def predict_next_word(model, tokenizer, input_text, max_sequence_len):
    # Tokenizing the input text
    tokenlist = tokenizer.texts_to_sequences([input_text])[0]

    # If the tokenlist is longer than max_sequence_len, truncate it
    if len(tokenlist) >= max_sequence_len:
        tokenlist = tokenlist[-(max_sequence_len - 1):]

    # Padding the sequence to the required length
    tokenlist = pad_sequences([tokenlist], maxlen=max_sequence_len - 1, padding='pre')

    # Predicting the next word
    predicted = model.predict(tokenlist, verbose=0)

    # Get the index with the highest probability
    predicted_index = predicted.argmax(axis=-1)

    # Convert index to word using the tokenizer's word index
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

# Streamlit app
st.set_page_config(page_title="Next Word Prediction", page_icon="📝", layout="wide")
st.title("Next Word Prediction of the Sentence using GRU and Early Stopping")

# Input section
input_text = st.text_input('Enter the sequence of words', 'Prajwal Patel is a good')

# Create a button to predict the next word
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    
    # Display the result
    st.success(f"The next word is: **{next_word}**")

# Additional Features
st.sidebar.header("History")
if 'history' not in st.session_state:
    st.session_state.history = []

if st.button("Add to History"):
    st.session_state.history.append((input_text, next_word))
    st.sidebar.write(f"Added: **{input_text} => {next_word}**")

# Show prediction history
if st.session_state.history:
    st.sidebar.subheader("Prediction History")
    for item in st.session_state.history:
        st.sidebar.write(f"{item[0]} => {item[1]}")

# Add custom CSS for better visuals
st.markdown("""
<style>
h1 {
    color: #4B0082;
}
.stButton>button {
    background-color: #FF5733;
    color: white;
    font-weight: bold;
    padding: 10px;
}
.stTextInput>div>input {
    border: 2px solid #4B0082;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)
