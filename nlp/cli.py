# -*- coding: utf-8 -*-

"""Demonstrating a very simple NLP project. Yours should be more exciting than this."""
import click
import glob
import pickle
import sys
import os

import numpy as np
import pandas as pd
import re
import requests
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import accuracy_score, classification_report

#from . import clf_path, config

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

@click.group()
def main(args=None):
    """Console script for nlp."""
    return 0

@main.command('web')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(port):
    """
    Launch the flask web app.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    
@main.command('dl-data')
def dl_data():
    """
    Download training/testing data.
    """
    # Load the PKU-SafeRLHF dataset for tuning the reward model. 
    print("Load the PKU-SafeRLHF dataset from huggingface.")
    train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    print("Finished loading")


#def data2df():
    #return pd.read_csv(config.get('data', 'file'))

@main.command('stats')
def stats():
    """
    Read the data files and print interesting statistics.
    """
    print("Load the PKU-SafeRLHF dataset from huggingface.")
    train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    print("The first 10 entries of the dataset:")
    print(train_dataset[:10]) 

@main.command('train')
def train():
    """
    Train a classifier and save it.
    """
    # (1) Read the data...
    os.system("python3 ../notebooks/cost_trainer.py")

'''
def do_cross_validation(clf, X, y):
    all_preds = np.zeros(len(y))
    for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X,y):
        clf.fit(X[train], y[train])
        all_preds[test] = clf.predict(X[test])
    print(classification_report(y, all_preds))    

def top_coef(clf, vec, labels=['liberal', 'conservative'], n=10):
    feats = np.array(vec.get_feature_names_out())
    print('top coef for %s' % labels[1])
    for i in np.argsort(clf.coef_[0])[::-1][:n]:
        print('%20s\t%.2f' % (feats[i], clf.coef_[0][i]))
    print('\n\ntop coef for %s' % labels[0])
    for i in np.argsort(clf.coef_[0])[:n]:
        print('%20s\t%.2f' % (feats[i], clf.coef_[0][i]))
'''
if __name__ == "__main__":
    sys.exit(main())
