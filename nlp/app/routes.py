from flask import render_template, flash, redirect, session
import torch
from . import app
from .forms import MyForm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
#from .. import clf_path

#import pickle
#import sys

#clf, vec = pickle.load(open(clf_path, 'rb'))
#print('read clf %s' % str(clf))
#print('read vec %s' % str(vec))
#labels = ['liberal', 'conservative']

device = 0 if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained("cost_model", num_labels=1, torch_dtype=torch.bfloat16)
model.cuda()

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	form = MyForm()
	result = None
	
	if form.validate_on_submit():
		input_field = form.input_field.data
		tokenized = tokenizer(input_field)
		cost = model(input_ids=torch.tensor(tokenized["input_ids"]).unsqueeze(0).to(device), attention_mask=torch.tensor(tokenized["attention_mask"]).unsqueeze(0).to(device))[0]
		print(cost)
		# flash(input_field)
		return render_template('myform.html', title='', form=form, 
								prediction=str(cost), confidence='1')
		#return redirect('/index')
	
	return render_template('myform.html', title='', form=form, prediction=None, confidence=None)
