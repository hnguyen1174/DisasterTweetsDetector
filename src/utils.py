from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import random
import numpy as np
import torch

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


def tokenize(batch):
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