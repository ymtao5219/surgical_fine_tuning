import argparse
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer, AutoModelForMultipleChoice, AutoModelForQuestionAnswering
#TODO: added for roberta
from transformers import AutoTokenizer, RobertaForSequenceClassification
import time 

import evaluate
import numpy as np

import yaml 

from data_loader import *
from utils import *

import torch

import logging
# Set logging level
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(args):
    set_random_seed(42)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")
        
    model_name = args.parent_model
    freeze_layers = args.freeze_layers
    task_name = args.task_name
    
    if task_name == "cola":
        metric = evaluate.load("matthews_correlation")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

    elif task_name == "stsb":
        metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

    else: 
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)


    if args.few_shot:
        data_loader = GlueDataloader(task_name, model_name)
        train_dataset, val_dataset = data_loader.get_train_val_split(args.few_shot)
    else: 
        data_loader = GlueDataloader(task_name, model_name)
        train_dataset, val_dataset = data_loader.get_train_val_split()

    # ipdb.set_trace()
    # Model
    if task_name == "copa": 
        model = AutoModelForMultipleChoice.from_pretrained(model_name)
    elif task_name == "multirc":
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    elif task_name == "stsb":
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
    else: 
        #TODO: changed for roberta
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(train_dataset.unique("label")))
        # model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(train_dataset.unique("label")))
    # ipdb.set_trace()
    def add_prefix(val):
        # TODO: changed for roberta
        # return "bert.encoder.layer." + str(val)
        return "roberta.encoder.layer." + str(val)

    # print("layers to freeze", freeze_layers)
    if freeze_layers:
        for i in range(len(freeze_layers)):
            freeze_layers[i] = add_prefix(freeze_layers[i])
        freeze_layers = tuple(freeze_layers)
        print("layers to freeze", freeze_layers)
        for name, param in model.named_parameters():
            if name.startswith(freeze_layers):
                param.requires_grad = False

    if args.benchmark == "glue":
        config_path = "configs/glue_training_args.yaml"
    else: 
        config_path = "configs/superglue_training_args.yaml"
        
    with open(config_path, "r") as f:
        training_args_dict = yaml.safe_load(f)
        
    # Training arguments
    training_args = TrainingArguments(**training_args_dict)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                               'attention_mask': torch.stack([f[1] for f in data]),
                               'labels': torch.tensor([f[2] for f in data])},
        compute_metrics=compute_metrics
    )

    # Record the start time
    start_time = time.time()

    # Train the model
    trainer.train()

    # Record the end time
    end_time = time.time()

    # Calculate the total training time
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    # print("evaluating on test set")
    # GLUE benchmark has not labels for test set, so the following code is commented out
    # # Evaluate the model
    # test_results = trainer.evaluate()
    # print(test_results)
    
    # Save the best model
    trainer.save_model(f"checkpoints/best_model_{args.task_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning a parent model")
    parser.add_argument("--parent_model", type=str, default="bert-base-cased", help="Name of the parent model to use from Hugging Face")
    parser.add_argument("--benchmark", type=str, default="glue", help="Name of the benchmark to use (glue or superglue)")
    parser.add_argument("--task_name", type=str, default="wic", help="Name of the task in GLUE/SuperGLUE to fine-tune on")
    parser.add_argument("--freeze_layers", nargs='+', type=int, help="List of which layers to freeze")
    parser.add_argument("--few_shot", type=int, help="Number of examples per class to use for fine-tuning")
    args = parser.parse_args()

    main(args)