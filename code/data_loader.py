from datasets import load_dataset
import evaluate
import numpy as np
import random
from transformers import AutoTokenizer

# import ipdb

class GlueDataloader:
    GLUE_TASKS = ["mrpc", "stsb", "rte", "wnli", "qqp", "mnli_mismatched", "mnli_matched", "qnli", "cola", "sst2" ]
    SUPERGLUE_TASKS = ["wic", "cb", "boolq", "copa", "multirc", "record", "wsc"]
    glue = "glue"
    super_glue = "super_glue"

    def __init__(self, task_name, model_name= "bert-base-cased", tokenizer=None):
        self.task_name = task_name.lower()
        self.model_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.task_type = ""
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if self.task_name in self.GLUE_TASKS:
            dataset = load_dataset(self.glue, self.task_name)
            self.task_type = self.glue
        elif self.task_name in self.SUPERGLUE_TASKS:
            dataset = load_dataset(self.super_glue, self.task_name)
            self.task_type = self.super_glue
        else:
            raise ValueError("Invalid task name. Please choose from GLUE or SuperGLUE tasks.")

        return dataset

    def get_samples(self, num_sentences, split="validation", seed=42):
        random.seed(seed)
        sampled_data = self.dataset[split].select(range(min(len(self.dataset[split]), num_sentences)))
        preprocess_function = self._get_preprocessing_function()
        sampled_data = sampled_data.map(preprocess_function)
        return sampled_data
    
    def get_train_val_split(self):
        
        # special case for mnli
        if self.task_name in ["mnli_mismatched", "mnli_matched"]:
            dataset_train_split = load_dataset(self.glue, "mnli", split="train")
            dataset_val_split = self.dataset["validation"]
        else:
            dataset_train_split = self.dataset["train"]
            dataset_val_split = self.dataset["validation"]
        
        train_dataset, val_dataset = dataset_train_split, dataset_val_split

        preprocess_function = self._get_preprocessing_function()

        # Preprocess datasets
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        val_dataset = val_dataset.map(preprocess_function, batched=True)
        return train_dataset, val_dataset
    
    def _get_preprocessing_function(self):
        if self.task_name in ["mrpc", "stsb", "rte", "wnli", "wic"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')
        elif self.task_name in ["qqp"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['question1'], examples['question2'], truncation=True, padding='max_length')
        elif self.task_name in ["mnli_mismatched", "mnli_matched", "cb"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')
        elif self.task_name in ["qnli"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['question'], examples['sentence'], truncation=True, padding='max_length')
        elif self.task_name in ["cola", "sst2"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['sentence'], truncation=True, padding='max_length')
        elif self.task_name in ["boolq"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['question'], examples['passage'], truncation=True, padding='max_length')
        elif self.task_name in ["copa"]:
            preprocess_function = lambda examples: self.tokenizer(examples['premise'], examples['choice1'], examples['choice2'], truncation=True, padding='max_length')
        elif self.task_name in ["multirc"]:
            preprocess_function = lambda examples: self.tokenizer(examples['paragraph'], examples['question'], examples['answer'], truncation=True, padding='max_length')
        elif self.task_name in ["record"]:
            preprocess_function = lambda examples: self.tokenizer(examples['passage'], examples['query'], truncation=True, padding='max_length')
        elif self.task_name in ["wsc"]:
            preprocess_function = lambda examples: self.tokenizer(examples['text'], truncation=True, padding='max_length')
        
        return preprocess_function
    
    def get_metric(self):
        def compute_metrics(eval_pred):
            metric = evaluate.load(self.task_type, self.task_name)
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        return compute_metrics


# example usage
# GLUE_TASKS = ["mrpc", "stsb", "rte", "wnli", "qqp", "mnli_mismatched", "mnli_matched", "qnli", "cola", "sst2" ]
# SUPERGLUE_TASKS = ["wic", "cb", "boolq", "copa", "multirc", "record", "wsc"]
# data = GlueDataloader("cola")
# samples = data.get_samples(10)
# train, val = data.get_train_val_split()
# ipdb.set_trace()