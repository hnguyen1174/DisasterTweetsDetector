from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import random
import numpy as np
import torch
import yaml
import torch
import torch.nn as nn
from transformers import BertModel
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

def preprocessing_for_bert(data, tokenizer, MAX_LEN):
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # tokenizer.encode_plus(): Tokenize and prepare for the model 
        # a sequence or a pair of sequences.
        encoded_sent = tokenizer.encode_plus(
            text=sent,    # Preprocess sentence
            add_special_tokens=True,          # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,               # Max length to truncate/pad
            pad_to_max_length=True,           # Pad sentence to max length
            # return_tensors='pt',            # Return PyTorch tensor
            truncation = True,                # Truncate to maximum length
            return_attention_mask=True        # Return attention mask
            )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

class BertweetClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertweetClassifier, self).__init__()
        
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        # self.LSTM = nn.LSTM(D_in,D_in,bidirectional=True)
        # self.clf = nn.Linear(D_in*2,2)

        # Instantiate an one-layer feed-forward classifier
        
        '''
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )
        '''

        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(D_in, D_out)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bertweet.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        # Feed input to BERT
        outputs = self.bertweet(input_ids=input_ids,
                                attention_mask=attention_mask)
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        hidden_state_dropout = self.dropout(last_hidden_state_cls)
        # Feed input to classifier to compute logits
        logits = self.classifier(hidden_state_dropout)

        return logits

def initialize_model(epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bertweet_classifier = BertweetClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bertweet_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bertweet_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bertweet_classifier, optimizer, scheduler