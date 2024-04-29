# This script is modified from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
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

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.1)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name:Optional[str] = field(
        default="PKU-Alignment/PKU-SafeRLHF",  # From https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
        metadata={
            "help": "The dataset used for training reward model. Default is PKU-SafeRLHF from https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF"
        },
    )
    train_split:Optional[str] = field(default="train", metadata={"help": "the split name of the dataset for trianing"})
    test_split:Optional[str] = field(default="test", metadata={"help": "the split name of the dataset for evaluating"})
    tokenizer_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default for your model",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=2,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=0,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=0,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "the maximum sequence length"})
    eval_first_step: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run eval after the first step"},
    )
    regularization: Optional[int] = field(
        default=0, 
        metadata={"help": "Extra regularization terms to the loss functions to get better generalizability and stabilize the training process"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# Load the PKU-SafeRLHF dataset for tuning the reward model. 
train_dataset = load_dataset(script_args.dataset_name, split=script_args.train_split)
# train_dataset = load_dataset("json", data_files="dataset/train.jsonl", split="train")
if script_args.train_subset > 0:
    train_dataset = train_dataset.select(range(script_args.train_subset))
eval_dataset = load_dataset(script_args.dataset_name, split=script_args.test_split)
# eval_dataset = load_dataset("json", data_files="dataset/test.jsonl", split="train")
if script_args.eval_subset > 0:
    eval_dataset = eval_dataset.select(range(script_args.eval_subset))


# Define the training args. Needs to be done before the model is loaded if using deepspeed.
model_name_split = script_args.tokenizer_name.split("/")[-1]
# The output dir name
output_name = (f"{model_name_split}_{script_args.dataset_name}_{script_args.train_subset}_{script_args.learning_rate}_cost")

training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="epoch",
    # eval_steps=5000,
    save_strategy="steps",
    save_steps=500,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=["labels"],   # We need the safe sign to evaluate the metric
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name="cost_model_llama2"
)


# Define the tokenizer
tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = AutoModelForSequenceClassification.from_pretrained(script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16)
model = get_peft_model(model, peft_config)
model.cuda()


# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not script_args.gradient_checkpointing
num_proc = 24  # Can adjust to be higher if have more processors.
original_columns = train_dataset.column_names


# Turn the dataset into pairs of prompt + response, where text_j is the unsafer prompt + response and text_k is the other
# For each prompt + response, we also need to know if this is safe.
# safe sign is -1 for safe text and 1 for unsafe text
# Then tokenize the dataset
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
        "safe_sign_j_k": []
    }

    for question, response_0, response_1, response_0_safe, response_1_safe, safer_response_id in zip(examples["prompt"], examples["response_0"] , examples["response_1"], examples["is_response_0_safe"], examples["is_response_1_safe"], examples["safer_response_id"]):
        if safer_response_id == 1:   # Safer response has lower cost, j refers to higher cost
            tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_0, truncation=True)
            tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_1, truncation=True)
            safe_sign_j = -1 if response_0_safe == True else 1     # return -1 for safe text and 1 for unsafe text
            safe_sign_k = -1 if response_1_safe == True else 1
        else:
            tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_1, truncation=True)
            tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_0, truncation=True)
            safe_sign_j = -1 if response_1_safe == True else 1     # return -1 for safe text and 1 for unsafe text
            safe_sign_k = -1 if response_0_safe == True else 1

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])
        new_examples["safe_sign_j_k"].append([safe_sign_j, safe_sign_k])


    return new_examples

# preprocess the dataset and filter out QAs that are longer than script_args.max_length
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= script_args.max_length and len(x["input_ids_k"]) <= script_args.max_length
)


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class CostDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "labels": torch.tensor([f["safe_sign_j_k"] for f in features]),
            "return_loss": True,
        }
        return batch
    

# Define the metric that we'll use for validation.
# Cannot use evaluate online, so I clone the repository and use it offline as described at https://github.com/huggingface/evaluate/issues/315
accuracy = evaluate.load("evaluate/metrics/accuracy/accuracy.py")

# safe sign is -1 for safe text and 1 for unsafe text
def compute_metrics(eval_pred):
    original_predictions, sign_labels = eval_pred
    # Here, predictions is costs_j and costs_k.
    # We want to see how much of the time costs_j > costs_k.
    predictions = np.argmax(original_predictions, axis=0)
    labels = np.zeros(predictions.shape)
    
    accuracy_dict = accuracy.compute(predictions=predictions, references=labels)

    # safe samples are supposed to have negative costs
    # unsafe samples are supposed to have positive costs
    predicted_sign = np.sign(np.concatenate(original_predictions, axis=0)).astype(np.int32)
    labels_sign = np.vstack((sign_labels[:,0].reshape(-1,1),sign_labels[:,1].reshape(-1,1)))
    sign_accuracy = accuracy.compute(predictions=predicted_sign, references=labels_sign)
    
    accuracy_dict["sign accuracy"] = sign_accuracy["accuracy"]

    return accuracy_dict


class CostTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        costs_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        costs_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        safe_sign_j = inputs["labels"][:,0].unsqueeze(dim=1)
        safe_sign_k = inputs["labels"][:,1].unsqueeze(dim=1)
        loss = -nn.functional.logsigmoid(costs_j - costs_k).mean() - nn.functional.logsigmoid(costs_j*safe_sign_j).mean() - nn.functional.logsigmoid(costs_k*safe_sign_k).mean() + script_args.regularization * (torch.square(costs_j).mean() + torch.square(costs_k).mean())
        if return_outputs:
            return loss, {"costs_j": costs_j, "costs_k": costs_k}
        return loss


# Train the model.
trainer = CostTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=CostDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
)


if script_args.eval_first_step:
    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True

    trainer.add_callback(EvaluateFirstStepCallback())

trainer.train(script_args.resume_from_checkpoint)

print("Saving last checkpoint of the model")
model.save_pretrained(output_name + "_peft_last_checkpoint")