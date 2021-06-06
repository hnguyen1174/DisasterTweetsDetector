import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from utils import *

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

    

