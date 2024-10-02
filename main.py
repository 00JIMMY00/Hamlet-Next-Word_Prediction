import tensorflow
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gdown

# Function to download and cache the model file
@st.cache_data
def download_model():
    url = "https://drive.google.com/uc?id=1LFESeIf3CgMD66s5Bb9qCa-tLDT6tN7y"
    output = 'next_word_lstmv2.h5'
    gdown.download(url, output, quiet=False, fuzzy=True)
    return output

# Function to load the model from the cached file
@st.cache_data
def load_lstm_model(model_path):
    return load_model(model_path)

# Function to download and cache the tokenizer (pickle file)
@st.cache_data
def download_tokenizer():
    url = "https://drive.google.com/uc?id=1wpSl-O9rrR46gJofPsVy5418aOvRlWMg"
    output = 'tokenizer.pickle'
    gdown.download(url, output, quiet=False, fuzzy=True)
    return output

# Function to load the tokenizer from the cached pickle file
@st.cache_data
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        return pickle.load(handle)

# Load the LSTM model (downloaded if not cached)
model_path = download_model()
model = load_lstm_model(model_path)

# Load the tokenizer (downloaded if not cached)
tokenizer_path = download_tokenizer()
tokenizer = load_tokenizer(tokenizer_path)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title("Hamlet Next Word Prediction With LSTM ")
input_text = st.text_input("Enter the sequence of Words", "To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')
