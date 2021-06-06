from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import pandas as pd
from utils import *
from transformers import Trainer

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = 2
    model_name = 'garynguyen1174/disaster_tweet_bert'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test = pd.read_csv('../data/test.csv')
    test_texts = test['text'].to_list()
    
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = TweetDataset(test_encodings)
    test_trainer = Trainer(model) 
    
    raw_pred, _, _ = test_trainer.predict(test_dataset) 
    y_pred = np.argmax(raw_pred, axis=1)
    
    preds = pd.DataFrame()
    preds['text'] = test['text']
    preds['pred'] = y_pred
    preds.to_csv('../data/preds.csv')


