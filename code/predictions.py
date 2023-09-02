import argparse
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer, AutoModelForMultipleChoice, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, RobertaForSequenceClassification
import pickle 
from datasets import load_metric
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
    # freeze_layers = args.freeze_layers
    load_model = args.load_model
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
            predictions = logits[:, 0]
            return metric.compute(predictions=predictions, references=labels)

    elif task_name=="record":
        metric = load_metric('super_glue', task_name)
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions[:, 0]
            return metric.compute(predictions=predictions, references=labels)

    else: 
        metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)


    if task_name == "stsb":
        def get_predictions(output):
            predictions = output.logits[:, 0].cpu().numpy()
            return predictions

    else: 
        def get_predictions(output):
            predictions = np.argmax(output.logits.cpu().numpy(), axis=-1)
            return predictions

    # Define a function to extract activations from the model
    def get_activations(model, inputs):
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            activations = outputs.hidden_states
        return activations
        

    if args.few_shot:
        data_loader = GlueDataloader(task_name, model_name)
        train_dataset, val_dataset = data_loader.get_train_val_split(args.few_shot)
    else: 
        data_loader = GlueDataloader(task_name, model_name)
        train_dataset, val_dataset = data_loader.get_train_val_split()

    # ipdb.set_trace()
    # Model
    if task_name == "copa": 
        model = AutoModelForMultipleChoice.from_pretrained(load_model)
    elif task_name == "multirc":
        model = AutoModelForQuestionAnswering.from_pretrained(load_model)
    elif task_name == "stsb":
        model = BertForSequenceClassification.from_pretrained(load_model, num_labels=1)
    else: 
        model = BertForSequenceClassification.from_pretrained(load_model, num_labels=len(train_dataset.unique("label")))
        # TODO: changed for roberta
        # model = RobertaForSequenceClassification.from_pretrained(load_model, num_labels=len(train_dataset.unique("label")))
    # ipdb.set_trace()
    def add_prefix(val):
        return "bert.encoder.layer." + str(val)
         # TODO: changed for roberta
        # return "roberta.encoder.layer." + str(val)

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
        compute_metrics=compute_metrics
    )

    activations = []
    predictions = []

    for batch in trainer.get_eval_dataloader():
        inputs = {k: v.to(trainer.args.device) for k, v in batch.items() if k != "label"}
        batch_activations = get_activations(model, inputs)
        activations.append(batch_activations)
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_predictions = get_predictions(outputs)
            predictions.append(batch_predictions)

    print("Predictions: \n", predictions)

    activations_file = 'results/logs/Pickle_files/' + task_name + '.pickle'
    with open(activations_file, 'wb') as f:
        pickle.dump(activations, f, protocol=pickle.HIGHEST_PROTOCOL)

    # To load these activations use the following code:
    # with open(activations_file, 'rb') as f:
    #     loaded_activations = pickle.load(f)
 

    print("Evaluating on validation set")
    # GLUE benchmark has not labels for test set, so the following code is commented out
    # Evaluate the model
    test_results = trainer.evaluate()
    print("\nValidation results: \n", test_results)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning a parent model")
    parser.add_argument("--parent_model", type=str, default="bert-base-cased", help="Name of the parent model to use from Hugging Face")
    parser.add_argument("--load_model", type=str, default="checkpoints/best_model_wsc", help="Name of the model to use to get predictions")
    parser.add_argument("--benchmark", type=str, default="super_glue", help="Name of the benchmark to use (glue or superglue)")
    parser.add_argument("--task_name", type=str, default="wsc", help="Name of the task in GLUE/SuperGLUE to fine-tune on")
    parser.add_argument("--few_shot", type=int, help="Number of examples per class to use for fine-tuning")
    args = parser.parse_args()

    main(args)