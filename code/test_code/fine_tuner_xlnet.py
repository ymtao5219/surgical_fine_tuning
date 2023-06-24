import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import XLNetForSequenceClassification, XLNetTokenizer, AdamW
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import argparse
from transformers import BertForSequenceClassification, AutoModelForMultipleChoice, AutoModelForQuestionAnswering
import time 

import evaluate
import numpy as np

import yaml
from code.test_code.data_loader_xlnet import XLNetGlueDataloader 

from code.test_code.data_loader_test import *
from utils import *

import torch

import logging
# Set logging level
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# import ipdb

# def main(args):
#     set_random_seed(42)
    
#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs")
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using {device}")
        
#     model_name = args.parent_model
#     freeze_layers = args.freeze_layers
#     task_name = args.task_name
    
#     if task_name == "cola":
#         metric = evaluate.load("matthews_correlation")

#         def compute_metrics(eval_pred):
#             logits, labels = eval_pred
#             predictions = np.argmax(logits, axis=-1)
#             return metric.compute(predictions=predictions, references=labels).to(device)

#     elif task_name == "stsb":
#         metric = evaluate.load("accuracy")
#         def compute_metrics(eval_pred):
#             logits, labels = eval_pred
#             predictions = logits[:, 0]
#             return metric.compute(predictions=predictions, references=labels).to(device)

#     else: 
#         metric = evaluate.load("accuracy")

#         def compute_metrics(eval_pred):
#             logits, labels = eval_pred
#             predictions = np.argmax(logits, axis=-1)
#             return metric.compute(predictions=predictions, references=labels).to(device)

#     data_loader = XLNetGlueDataloader(task_name, model_name)

#     # Model
#     if task_name == "copa": 
#         model = AutoModelForMultipleChoice.from_pretrained(model_name)
#     elif task_name == "multirc":
#         model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#     elif task_name == "stsb":
#         model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=1)
#     else: 
#         model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=len(data_loader.dataset["train"].unique("label")))
        
#     def add_prefix(val):
#         return "transformer.layer." + str(val)


#     # print("layers to freeze", freeze_layers)
#     if freeze_layers:
#         for i in range(len(freeze_layers)):
#             freeze_layers[i] = add_prefix(freeze_layers[i])
#         freeze_layers = tuple(freeze_layers)
#         print("layers to freeze", freeze_layers)
#         for name, param in model.named_parameters():
#             if name.startswith(freeze_layers):
#                 param.requires_grad = False

#     if args.benchmark == "glue":
#         config_path = "configs/glue_training_args.yaml"
#     else: 
#         config_path = "configs/superglue_training_args.yaml"
        
#     with open(config_path, "r") as f:
#         training_args_dict = yaml.safe_load(f)
        
#     # Training arguments
#     input_ids = []
#     attention_masks = []
#     labels = []

#     for example in data_loader.dataset["train"]:
#         # print(example)
#         inputs = data_loader.tokenizer.encode_plus(
#             example["premise"],
#             example["hypothesis"],
#             add_special_tokens=True,
#             max_length=128,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#         input_ids.append(inputs["input_ids"].squeeze())
#         attention_masks.append(inputs["attention_mask"].squeeze())
#         labels.append(example["label"])

#     input_ids = torch.stack(input_ids)
#     attention_masks = torch.stack(attention_masks)
#     labels = torch.tensor(labels)

#     # Create data loader for training
#     train_dataset = TensorDataset(input_ids, attention_masks, labels)
#     train_data_loader = DataLoader(train_dataset, batch_size=training_args_dict['per_device_train_batch_size'], shuffle=True)

#     # Create optimizer
#     optimizer = AdamW(model.parameters(), lr=training_args_dict['learning_rate'])

#     # Fine-tuning loop
#     # Record the start time
#     start_time = time.time()

#     model.train()
#     for epoch in range(training_args_dict['num_train_epochs']):
#         total_loss = 0
#         for batch in train_data_loader:
#             input_ids = batch[0].to(device)
#             attention_mask = batch[1].to(device)
#             labels = batch[2].to(device)
            
#             optimizer.zero_grad()
            
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             logits = outputs.logits
            
#             total_loss += loss.item()
            
#             loss.backward()
#             optimizer.step()
        
#         average_loss = total_loss / len(train_data_loader)
#         print(f"Epoch: {epoch+1}, Average Loss: {average_loss}")

#     # Step 4: Evaluate the fine-tuned model
#     # Preprocess the evaluation dataset into input features
#     eval_input_ids = []
#     eval_attention_masks = []
#     eval_labels = []

#     for example in data_loader.dataset["validation"]:
#         inputs = data_loader.tokenizer.encode_plus(
#             example["premise"],
#             example["hypothesis"],
#             add_special_tokens=True,
#             max_length=128,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
    
#         eval_input_ids.append(inputs["input_ids"].squeeze())
#         eval_attention_masks.append(inputs["attention_mask"].squeeze())
#         eval_labels.append(example["label"])

#     eval_input_ids = torch.stack(eval_input_ids)
#     eval_attention_masks = torch.stack(eval_attention_masks)
#     eval_labels = torch.tensor(eval_labels)

#     # Create data loader for evaluation
#     eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_labels)
#     eval_data_loader = DataLoader(eval_dataset, batch_size=training_args_dict['per_device_eval_batch_size'], shuffle=False)

#     model.eval()
#     predictions = []
#     true_labels = []
#     for batch in eval_data_loader:
#         input_ids = batch[0].to(device)
#         attention_mask = batch[1].to(device)
#         labels = batch[2].to(device)
        
#         with torch.no_grad():
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
#         logits = outputs.logits
#         _, predicted_labels = torch.max(logits, dim=1)
        
#         predictions.extend(predicted_labels.cpu().tolist())
#         true_labels.extend(labels.tolist())

#     accuracy = compute_metrics(true_labels, predictions)
#     # accuracy = accuracy_score(true_labels, predictions)
#     print("Evaluation Accuracy:", accuracy)


#     # Train the model
#     # trainer.train()

#     # Record the end time
#     end_time = time.time()

#     # Calculate the total training time
#     training_time = end_time - start_time
#     print(f"Total training time: {training_time:.2f} seconds")

#     # print("evaluating on test set")
#     # GLUE benchmark has not labels for test set, so the following code is commented out
#     # # Evaluate the model
#     # test_results = trainer.evaluate()
#     # print(test_results)
    
#     # Save the best model
#     # trainer.save_model(f"checkpoints/best_model_{args.task_name}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Fine-tuning a parent model")
#     parser.add_argument("--parent_model", type=str, default="xlnet-base-cased", help="Name of the parent model to use from Hugging Face")
#     parser.add_argument("--benchmark", type=str, default="super_glue", help="Name of the benchmark to use (glue or superglue)")
#     parser.add_argument("--task_name", type=str, default="cb", help="Name of the task in GLUE/SuperGLUE to fine-tune on")
#     parser.add_argument("--freeze_layers", nargs='+', type=int, help="List of which layers to freeze")
#     parser.add_argument("--few_shot", type=int, help="Number of examples per class to use for fine-tuning")
#     args = parser.parse_args()

#     main(args)



# def main(args):
    
# Step 1: Load the CB dataset using Hugging Face load_dataset
dataset = load_dataset("super_glue", "cb")

# Step 2: Prepare the XLNet model and tokenizer
model_name = "xlnet-base-cased"
model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = XLNetTokenizer.from_pretrained(model_name)

# Step 3: Fine-tune the XLNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define training hyperparameters
epochs = 3
learning_rate = 2e-5
batch_size = 32

# Preprocess the dataset into input features
input_ids = []
attention_masks = []
labels = []

for example in dataset["train"]:
    inputs = tokenizer.encode_plus(
        example["premise"],
        example["hypothesis"],
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids.append(inputs["input_ids"].squeeze())
    attention_masks.append(inputs["attention_mask"].squeeze())
    labels.append(example["label"])

input_ids = torch.stack(input_ids)
attention_masks = torch.stack(attention_masks)
labels = torch.tensor(labels)

# Create data loader for training
train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Fine-tuning loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_data_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    average_loss = total_loss / len(train_data_loader)
    print(f"Epoch: {epoch+1}, Average Loss: {average_loss}")

# Step 4: Evaluate the fine-tuned model
# Preprocess the evaluation dataset into input features
eval_input_ids = []
eval_attention_masks = []
eval_labels = []

for example in dataset["validation"]:
    inputs = tokenizer.encode_plus(
        example["premise"],
        example["hypothesis"],
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    eval_input_ids.append(inputs["input_ids"].squeeze())
    eval_attention_masks.append(inputs["attention_mask"].squeeze())
    eval_labels.append(example["label"])

eval_input_ids = torch.stack(eval_input_ids)
eval_attention_masks = torch.stack(eval_attention_masks)
eval_labels = torch.tensor(eval_labels)

# Create data loader for evaluation
eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_labels)
eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

model.eval()
predictions = []
true_labels = []
for batch in eval_data_loader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    labels = batch[2]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    _, predicted_labels = torch.max(logits, dim=1)
    
    predictions.extend(predicted_labels.cpu().tolist())
    true_labels.extend(labels.tolist())

accuracy = accuracy_score(true_labels, predictions)
print("Evaluation Accuracy:", accuracy)


