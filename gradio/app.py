import numpy as np
import pandas as pd
import tensorflow as tf
import gradio as gr
import nltk
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

model = load_model("mr_robot.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
maxlen = 241
input_dim = 387

stop_words = set(stopwords.words("english"))

def clean(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    text = text.strip()
    return text

def predict(content):
    context_clean = clean(content)
    test = tokenizer.texts_to_sequences([context_clean])
    test = pad_sequences(test, maxlen=maxlen)
    res = model.predict(test)
    res = np.argmax(res)
    return res

if __name__ == "__main__":
    iface = gr.Interface(fn=predict, inputs=gr.inputs.Textbox(label="Write text here..."), outputs=gr.outputs.Textbox(label="Result"))
    iface.launch(share=False, debug=True)