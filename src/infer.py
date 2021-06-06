from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, pipeline
import pandas as pd
import numpy as np
import torch
import sys

if __name__ == '__main__':

    text = str(sys.argv[0])

    model_name = 'garynguyen1174/disaster_tweet_bert'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_tensor = tokenizer.encode(text, return_tensors='pt')
    logits = model(input_tensor)[0]
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(logits)[0][1]
    probs = probs.cpu().detach().numpy()

    print('Probability of disaster tweet is {}%.'.format(round(probs*100, 2)))