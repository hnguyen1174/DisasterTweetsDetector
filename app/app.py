import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
import torch
import codecs
from bs4 import BeautifulSoup

#---------------------------------#
word_attributions = None

#---------------------------------#
# Page layout
# Page expands to full width
st.set_page_config(page_title='Disaster Tweet Detector',
    layout='wide')

#---------------------------------#
st.write("""
# Disaster Tweet Detector
This application is built to detect whether a tweet signals true disasters.
""")

#---------------------------------#
# Main panel

# Displays the dataset
user_input = st.text_input(
    "Please input Tweet:", 
    "Sample text."
    )


if user_input is not None and st.button('Compute'):
    
    model_name = 'garynguyen1174/disaster_tweet_bert'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_tensor = tokenizer.encode(user_input, return_tensors='pt')
    logits = model(input_tensor)[0]
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(logits)[0][1]
    probs = probs.cpu().detach().numpy()

    st.write('Probability of disaster tweet is:')
    st.info('{}%'.format(round(probs*100, 2)))

if st.sidebar.button('Intepret') and user_input is not None:
   
    st.write('Intepret the Model:')
    model_name = 'garynguyen1174/disaster_tweet_bert'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    cls_explainer = SequenceClassificationExplainer(
        model,
        tokenizer
        )
    word_attributions = cls_explainer(user_input)
    cls_explainer.visualize("viz.html")
    f = codecs.open("viz.html", 'r', 'utf-8')
    document = f.read()
    st.components.v1.html(document)