from datasets import load_dataset, Dataset, DatasetDict
import random
from transformers import AutoTokenizer
import torch
# import ipdb

class GlueDataloader:
    GLUE_TASKS = ["mrpc", "stsb", "rte", "wnli", "qqp", "mnli_mismatched", "mnli_matched", "qnli", "cola", "sst2" ]
    SUPERGLUE_TASKS = ["wic", "cb", "boolq", "copa", "multirc", "record", "wsc"]

    def __init__(self, task_name, model_name= "bert-base-cased", tokenizer=None):
        self.task_name = task_name.lower()
        self.model_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if self.task_name in self.GLUE_TASKS:
            dataset = load_dataset("glue", self.task_name)
        elif self.task_name in self.SUPERGLUE_TASKS:
            dataset = load_dataset("super_glue", self.task_name)
        else:
            raise ValueError("Invalid task name. Please choose from GLUE or SuperGLUE tasks.")

        return dataset

    def get_samples(self, num_sentences, split="validation", seed=42):
        random.seed(seed)
        sampled_data = self.dataset[split].select(range(min(len(self.dataset[split]), num_sentences)))
        preprocess_function = self._get_preprocessing_function()

        sampled_data = sampled_data.map(preprocess_function)
        
        return sampled_data

    def get_train_val_split(self, num_samples_per_class=None):
        
        # special case for mnli
        if self.task_name in ["mnli_mismatched", "mnli_matched"]:
            dataset_train_split = load_dataset("glue", "mnli", split="train")
            dataset_val_split = self.dataset["validation"]
        else:
            dataset_train_split = self.dataset["train"]
            dataset_val_split = self.dataset["validation"]
        
        train_dataset, val_dataset = dataset_train_split, dataset_val_split
        preprocess_function = self._get_preprocessing_function()
        
        if num_samples_per_class is None: 
            # Preprocess datasets
            train_dataset = train_dataset.map(preprocess_function, batched=True)
            val_dataset = val_dataset.map(preprocess_function, batched=True)
            return train_dataset, val_dataset
        
        else: 
            # Check if the dataset has the 'label' field
            if "label" not in train_dataset.column_names:
                raise ValueError(f"Task '{self.task_name}' does not have a 'label' field in the dataset.")

            # Get the unique class labels
            unique_labels = set(train_dataset["label"])

            # Get the desired number of samples per class using filters
            selected_samples = []
            for label in unique_labels:
                samples_with_label = train_dataset.filter(lambda x: x["label"] == label)

                if len(samples_with_label) < num_samples_per_class:
                    print(f"Warning: Task '{self.task_name}' has less than {num_samples_per_class} samples for class {label}.")
                    selected_samples.extend(samples_with_label)
                else:
                    selected_samples.extend(random.sample(list(samples_with_label), num_samples_per_class))

            
            # Create a dictionary containing the selected samples
            selected_samples_dict = {key: [sample[key] for sample in selected_samples] for key in train_dataset.column_names}

            # Create a new dataset from the selected samples using Dataset.from_dict()
            selected_dataset = Dataset.from_dict(selected_samples_dict)

            # Preprocess the sampled training and validation datasets
            train_dataset = selected_dataset.map(preprocess_function, batched=True)
            val_dataset = val_dataset.map(preprocess_function, batched=True)

            return train_dataset, val_dataset
    
    def _get_preprocessing_function(self):
        '''
        Get the preprocessing function for each task
        '''
        # GLUE tasks
        if self.task_name in ["mrpc", "stsb", "rte", "wnli"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')
        elif self.task_name in ["qqp"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['question1'], examples['question2'], truncation=True, padding='max_length')
        elif self.task_name in ["mnli_mismatched", "mnli_matched"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')
        elif self.task_name in ["qnli"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['question'], examples['sentence'], truncation=True, padding='max_length')
        elif self.task_name in ["cola", "sst2"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['sentence'], truncation=True, padding='max_length')
        
        # SuperGLUE tasks
        elif self.task_name in ["boolq"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['question'], examples['passage'], truncation=True, padding='max_length')
        elif self.task_name in ["cb"]:
            preprocess_function = lambda examples: self.tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')
        elif self.task_name in ["copa"]:
            def preprocess_function_copa(examples):
                # Combine the premise and choices into pairs of sentences
                sentences = []
                for premise, choice1, choice2 in zip(examples['premise'], examples['choice1'], examples['choice2']):
                    sentences.append((premise, choice1))
                    sentences.append((premise, choice2))

                # Tokenize the pairs of sentences
                encoded = self.tokenizer(
                    [sent[0] for sent in sentences],
                    [sent[1] for sent in sentences],
                    truncation=True,
                    padding='max_length',
                    max_length=512
                )

                # Duplicate each label to match the pairs of sentences
                labels = [label * 2 for label in examples['label']]

                # Calculate the difference between the logits of the two choices
                encoded['labels'] = [1 if i % 2 == label else 0 for i, label in enumerate(labels)]

                return encoded
            preprocess_function = preprocess_function_copa
        elif self.task_name in ["multirc"]:
            def preprocess_data_multirc(examples):
                # Create a list to store the sentence pairs
                sentence_pairs = []
                
                # Iterate over the examples and create pairs of passage, question, and answer
                for passage, questions, answers, labels in zip(examples['passage'], examples['questions'], examples['answers'], examples['labels']):
                    for question, answer, label in zip(questions, answers, labels):
                        sentence_pairs.append((passage, question, answer))

                # Tokenize the pairs of sentences
                encoded = self.tokenizer(
                    [f"{pair[0]} {pair[1]}" for pair in sentence_pairs],
                    [pair[2] for pair in sentence_pairs],
                    truncation=True,
                    padding='max_length',
                    max_length=512
                )

                # Use the provided labels
                encoded['labels'] = [label for labels in examples['labels'] for label in labels]

                return encoded
            preprocess_function = preprocess_data_multirc
            
        elif self.task_name in ["record"]:
            def preprocess_data_record(examples):
                # Create a list to store the sentence pairs
                sentence_pairs = []
                
                # Iterate over the examples and create pairs of passage and question
                for passage, queries, entities, answers in zip(examples['passage'], examples['query'], examples['entities'], examples['answers']):
                    for query, entity, answer in zip(queries, entities, answers):
                        sentence_pairs.append((passage, query, entity))

                # Tokenize the pairs of sentences
                encoded = self.tokenizer(
                    [f"{pair[0]} {pair[1]}" for pair in sentence_pairs],
                    [pair[2] for pair in sentence_pairs],
                    truncation=True,
                    padding='max_length',
                    max_length=512
                )

                # Use the provided answers
                encoded['labels'] = [answer['start'] for answers in examples['answers'] for answer in answers]

                return encoded
            preprocess_function = preprocess_data_record
        elif self.task_name in ["wic"]:
            preprocess_function = lambda examples: self.tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')
        elif self.task_name in ["wsc"]:
            def preprocess_data_wsc(examples):
                # Replace the pronoun with a special token and create a list to store the sentence pairs
                sentence_pairs = []
                for text, span1, span2 in zip(examples['text'], examples['span1'], examples['span2']):
                    pronoun, antecedent = (span1, span2) if span1['start'] < span2['start'] else (span2, span1)
                    modified_text = text[:pronoun['start']] + self.tokenizer.mask_token + text[pronoun['end']:]
                    sentence_pairs.append((modified_text, antecedent['text']))

                # Tokenize the pairs of sentences
                encoded = self.tokenizer(
                    [pair[0] for pair in sentence_pairs],
                    [pair[1] for pair in sentence_pairs],
                    truncation=True,
                    padding='max_length',
                    max_length=512
                )

                # Use the provided labels
                encoded['labels'] = examples['label']

                return encoded
            preprocess_function = preprocess_data_wsc
        return preprocess_function
    
# example usage
# GLUE_TASKS = ["mrpc", "stsb", "rte", "wnli", "qqp", "mnli_mismatched", "mnli_matched", "qnli", "cola", "sst2" ]
# SUPERGLUE_TASKS = ["boolq", "cb", "copa", "multirc", "record", "wic",  "wsc"]
# data = GlueDataloader("cola")
# samples = data.get_samples(10)
# train, val = data.get_train_val_split()
# ipdb.set_trace()