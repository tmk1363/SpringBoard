#!/usr/bin/env python
# coding: utf-8

# In[17]:
# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI

# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI

from google.colab import drive
drive.mount('/content/drive')

#model loading 

source_folder = '/content/Data'
destination_folder = '/content/Data'

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="0.1",
)

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']



import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea

#loading torch model
best_model = BERT().to(device)

#best_model.load_state_dict(torch.load('/content/model.pt'))

load_checkpoint('/content/model.pt', best_model)

#best_model.eval()

# New Section









def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, titletext), _ in test_loader:

                labels = labels.type(torch.LongTensor)           
                labels = labels.to(device)
                titletext = titletext.type(torch.LongTensor)  
                titletext = titletext.to(device)
                output = model(titletext, labels)

                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])


@app.get("/predict-review")
def predict(review: str):
  # Model parameter
  MAX_SEQ_LEN = 128
  PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
  UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

  # Fields

  label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
  text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                    fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
  fields = [('label', label_field),('titletext', text_field)]

  # TabularDataset

  test1 = TabularDataset(path='/content/Data/test1.csv', format='CSV', fields=fields, skip_header=True)

  # Iterators

  test_iter1 = Iterator(test1, batch_size=16, device=device, train=False, shuffle=False, sort=False)
  Pred = evaluate(best_model,test_iter1)
  return Pred

predict("happy")

