import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments
import torch
from utils import *

if __name__ == '__main__':

    train = pd.read_csv('../data/train.csv')
    
    all_train_texts = train.text.to_list()
    all_train_labels = train.target.to_list()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_train_texts, all_train_labels, 
        test_size=0.2, 
        random_state=2021
    )

    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'bert-base-uncased'
    num_labels = 2

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset = TweetDataset(val_encodings, val_labels)

    model = (AutoModelForSequenceClassification
            .from_pretrained(model_name, num_labels=num_labels)
            .to(device))
    config = AutoConfig.from_pretrained(model_name)

    set_cuda_seed()

    training_args = TrainingArguments(
        output_dir='./results',         
        num_train_epochs=2,             
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=64,   
        warmup_steps=500,
        evaluation_strategy="epoch",
        weight_decay=0.01,               
        logging_dir='./logs',       
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,                      
        args=training_args,                 
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save Local
    trainer.save_model('../models/bert')
    tokenizer.save_pretrained('../models/bert/tokenizer')
    config.save_pretrained('../models/bert/tokenizer')

    # Save Huggingface Hub
    model.push_to_hub('disaster_tweet_bert')
    tokenizer.push_to_hub('disaster_tweet_bert')
    config.push_to_hub('disaster_tweet_bert')


    
