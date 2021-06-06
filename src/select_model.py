import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments
import torch
from utils import *
import itertools
from pprint import pprint
from run_bert import *

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

    # Create all combinations to iterate over
    model_names = [
        'bert-base-uncased',
        'roberta-base',
        'distilbert-base-uncased'
    ]

    train_epochs = [2, 3, 4]
    train_batch_size = [4, 8, 16, 32]

    options = [
        model_names,
        train_epochs,
        train_batch_size
    ]
    options = list(itertools.product(*options))
    option_flags = [{
        'model_name': i[0],
        'train_epochs': i[1],
        'train_batch_size': i[2],
        'eval_batch_size': 64,
        'num_labels': 2,
        'train_texts': train_texts,
        'val_texts': val_texts,
        'train_labels': train_labels,
        'val_labels': val_labels
    } for i in options]

    run_bert(**option_flags[0])



