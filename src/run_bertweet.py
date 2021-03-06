import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from utils import *
from transformers import AutoTokenizer, AutoModel, AutoConfig
import random
import time

def preprocessing_for_bert(data, tokenizer, MAX_LEN):
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # tokenizer.encode_plus(): Tokenize and prepare for the model 
        # a sequence or a pair of sequences.
        encoded_sent = tokenizer.encode_plus(
            text=sent,                        # Preprocess sentence
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

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

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

if __name__ == '__main__':

    config = load_config('config.yaml')
    train_df = pd.read_csv(config['train_path'])

    # Train-test split
    all_train_texts = train_df.text.to_list()
    all_train_labels = train_df.target.to_list()

    X_train, X_val, y_train, y_val = train_test_split(
        all_train_texts, all_train_labels, 
        test_size=0.2, 
        random_state=42
    )

    # Specify model name
    model_name = "vinai/bertweet-base"

    # Tokenizing data
    max_len = config['max_len']
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(X_train, tokenizer, max_len)
    val_inputs, val_masks = preprocessing_for_bert(X_val, tokenizer, max_len)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
    batch_size = config['training_batch_size']

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # Define model
    bertweet = AutoModel.from_pretrained(model_name)

    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    set_cuda_seed(42)    
    bertweet_classifier, optimizer, scheduler = initialize_model(epochs=3)
    train(bertweet_classifier, train_dataloader, val_dataloader, epochs=3, evaluation=True)

    # Save Local
    bert_config = AutoConfig.from_pretrained(model_name)

    model_path = '../models/bertweet'
    tokenizer_path = os.path.join(model_path, 'tokenizer')
    bertweet_classifier.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)
    bert_config.save_pretrained(tokenizer_path)

    # Save Huggingface Hub
    hub_model_path = 'disaster_tweet_bertweet'
    bertweet_classifier.push_to_hub(hub_model_path)
    tokenizer.push_to_hub(hub_model_path)
    bert_config.push_to_hub(hub_model_path)





