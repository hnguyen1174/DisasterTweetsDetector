from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import random
import numpy as np
import torch
import yaml
import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

def compute_metrics(pred):
    """
    This function computes metrics for Transformers' fine tuning
    
    Args:
        pred: predictions from Transformers' Trainer
    
    Returns:
        A dictionary that contains metrics of interest for binary classification:
            (1) Accuracy
            (2) Precision
            (3) Recall
            (4) F1 Score
            (5) AUC
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "auc": auc}


def tokenize(batch, tokenizer):
    """
    Tokenize by batches for Transformers
    """
    return tokenizer(batch["text"], padding=True, truncation=True)


def set_cuda_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def load_config(config_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config