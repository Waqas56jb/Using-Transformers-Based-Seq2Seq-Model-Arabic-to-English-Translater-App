import streamlit as st
import torch
import pickle
from transformers import MarianMTModel, MarianTokenizer

# Load the trained model
with open("nmt_model.pkl", "rb") as f:
    model = pickle.load(f)

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

# Streamlit UI
st.title("Arabic to English Translator")
st.write("Enter Arabic text and get the English translation.")

arabic_text = st.text_area("Enter Arabic Text:")

if st.button("Translate"):
    if arabic_text:
        # Tokenize and translate
        inputs = tokenizer(arabic_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            translated_ids = model.generate(**inputs)
        translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
        
        st.subheader("Translated English Text:")
        st.write(translated_text)
    else:
        st.warning("Please enter Arabic text.")
