import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments
import torch
from utils import *

def run_bert(model_name, train_epochs, train_batch_size, eval_batch_size,
             num_labels, train_texts, val_texts, train_labels, val_labels):

    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset = TweetDataset(val_encodings, val_labels)

    model = (AutoModelForSequenceClassification
        .from_pretrained(model_name, num_labels=num_labels)
        .to(device))

    set_cuda_seed()

    training_args = TrainingArguments(
        output_dir='./results',         
        num_train_epochs=train_epochs,             
        per_device_train_batch_size=train_batch_size,  
        per_device_eval_batch_size=eval_batch_size,   
        warmup_steps=500,
        evaluation_strategy='epoch',
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
    return model

if __name__ == '__main__':

    config = load_config('config.yaml')
    train = pd.read_csv(config['train_path'])
    
    all_train_texts = train.text.to_list()
    all_train_labels = train.target.to_list()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_train_texts, all_train_labels, 
        test_size=0.2, 
        random_state=2021
    )

    model_name = config['model']

    trained_model = run_bert(
        model_name = model_name,
        train_epochs = config['train_epoch'],
        train_batch_size = config['training_batch_size'],
        eval_batch_size = config['eval_batch_size'],
        num_labels = config['num_labels'],
        train_texts = train_texts,
        val_texts = val_texts,
        train_labels = train_labels,
        val_labels = val_labels
    )

    bert_config = AutoConfig.from_pretrained(model_name)
    
    # Save Local
    if config['save_local']:
        trained_model.save_pretrained(config['local_model_path'])
        tokenizer_path = os.path.join(config['local_model_path'], 'tokenizer')
        tokenizer.save_pretrained(tokenizer_path)
        bert_config.save_pretrained(tokenizer_path)

    # Save Huggingface Hub
    if config['push_hf_hub']:
        hub_model_path = config['hub_model_path']
        trained_model.push_to_hub(hub_model_path)
        tokenizer.push_to_hub(hub_model_path)
        bert_config.push_to_hub(hub_model_path)