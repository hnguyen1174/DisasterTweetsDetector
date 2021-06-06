import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from utils import *
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

if __name__ == '__main__':

    config = load_config('config.yaml')

    train = pd.read_csv(config['train_path'])
    
    all_train_texts = train.text.to_list()
    all_train_labels = train.target.to_list()

    X_train, X_val, y_train, y_val = train_test_split(
        all_train_texts, all_train_labels, 
        test_size=0.2, 
        random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

    MAX_LEN = 250

    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(X_train, tokenizer, MAX_LEN)
    val_inputs, val_masks = preprocessing_for_bert(X_val, tokenizer, MAX_LEN)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
    batch_size = 12

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")





