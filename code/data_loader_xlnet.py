from datasets import load_dataset, load_metric, Dataset, DatasetDict
import random
from transformers import AutoTokenizer, XLNetTokenizer
import torch
from collections import defaultdict
# import ipdb
from dataclasses import dataclass

class XLNetGlueDataloader:
    GLUE_TASKS = ["mrpc", "stsb", "rte", "wnli", "qqp", "mnli_mismatched", "mnli_matched", "qnli", "cola", "sst2" ]
    SUPERGLUE_TASKS = ["wic", "cb", "boolq", "copa", "multirc", "record", "wsc"]

    def __init__(self, task_name, model_name= "xlnet-base-cased", tokenizer=None):
        self.task_name = task_name.lower()
        self.model_name = model_name
        
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name, use_fast=True)
        
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if self.task_name in self.GLUE_TASKS:
            dataset = load_dataset("glue", self.task_name)
        elif self.task_name in self.SUPERGLUE_TASKS:
            dataset = load_dataset("super_glue", self.task_name)
        else:
            raise ValueError("Invalid task name. Please choose from GLUE or SuperGLUE tasks.")

        return dataset
  
    def _get_preprocessing_function(self):
        '''
        Get the preprocessing function for each task
        '''
        
        # GLUE tasks
        if self.task_name in ["mrpc", "stsb", "rte", "wnli"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)
        elif self.task_name in ["qqp"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['question1'], examples['question2'], truncation=True)
        elif self.task_name in ["mnli_mismatched", "mnli_matched"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['premise'], examples['hypothesis'], truncation=True)
        elif self.task_name in ["qnli"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['question'], examples['sentence'], truncation=True)
        elif self.task_name in ["cola", "sst2"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['sentence'], truncation=True)
        
        # SuperGLUE tasks
        elif self.task_name in ["boolq"]: 
            preprocess_function = lambda examples: self.tokenizer(examples['question'], examples['passage'], truncation=True)

        # NLI task: classification
        elif self.task_name in ["cb"]:
            # def preprocess_function_cb(examples):
            #     encoded = self.tokenizer(examples['premise'], examples['hypothesis'], truncation=True)
            #     encoded.update({"label": examples["label"]})
            #     return encoded
            preprocess_function = lambda examples: self.tokenizer(examples['premise'], examples['hypothesis'], truncation=True)

        # qa task 
        elif self.task_name in ["copa"]:
            def preprocess_function_copa(examples):
                # COPA_DICT = {"cause": "What was the cause of this?", "effect": "What happened as a result?",}

                # contexts = [p + " " + COPA_DICT[q] for p, q in zip(examples["premise"], examples["question"])]
                # sentences_a = [ctx + " " + choice for ctx, choice in zip(contexts, examples["choice1"])]
                # sentences_b = [ctx + " " + choice for ctx, choice in zip(contexts, examples["choice2"])]
                # encoded = self.tokenizer(
                #     sentences_a,
                #     sentences_b,
                #     truncation=True,
                #     padding="max_length",
                # )
                # encoded.update({"label": examples["label"]})
                
                # return encoded
                CONTEXT_COL = "premise"
                QUESTION_COL = "question"
                ANSWER_1_COL = "choice1"
                ANSWER_2_COL = "choice2"

                question_headers = examples[QUESTION_COL]
                first_sentences = [
                    [f"{examples[CONTEXT_COL][i]} What was the cause of this? "]*2 if header == "cause" else\
                    [f"{examples[CONTEXT_COL][i]} What was the effect of this? "]*2\
                        for i, header in enumerate(question_headers)
                ]
                first_sentences = sum(first_sentences, [])
                
                second_sentences = [
                    [examples[end][i] for end in [ANSWER_1_COL, ANSWER_2_COL]] for i, header in enumerate(question_headers)
                ]
                second_sentences = sum(second_sentences, [])
                tokenized_examples = self.tokenizer(first_sentences, second_sentences, truncation=True, padding="max_length")
                
                return {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
                        
            preprocess_function = preprocess_function_copa
        
        # qa task 
        elif self.task_name in ["multirc"]:
            def preprocess_data_multirc(examples):
                contexts = [paragraph + " " + question for paragraph, question in zip(examples["paragraph"], examples["question"])]
                encoded = self.tokenizer(
                contexts,
                examples["answer"],
                truncation=True,
                padding="max_length",
                )
                encoded.update({"label": examples["label"]})
                return encoded
            preprocess_function = preprocess_data_multirc
        
        # qa task 
        elif self.task_name in ["record"]:

            def preprocess_data_record(examples):
                encoded = defaultdict(list)
                for idx, passage, query, entities, answers in zip(
                    examples["idx"], examples["passage"], examples["query"], examples["entities"], examples["answers"]
                ):
                    for entity in entities:
                        label = 1 if entity in answers else 0
                        query_filled = query.replace("@placeholder", entity)
                        example_encoded = self.tokenizer(
                            passage,
                            query_filled,
                            truncation=True,
                            padding="max_length",
                            return_overflowing_tokens=True,
                        )
                        encoded["idx"].append(idx)
                        encoded["passage"].append(passage)
                        encoded["query"].append(query_filled)
                        encoded["entities"].append(entity)
                        encoded["answers"].append(answers)
                        encoded["input_ids"].append(example_encoded["input_ids"])
                        encoded["label"].append(label)
                        if "token_type_ids" in example_encoded:
                            encoded["token_type_ids"].append(example_encoded["token_type_ids"])
                        if "attention_mask" in example_encoded:
                            encoded["attention_mask"].append(example_encoded["attention_mask"])
                            
                return encoded
            
            preprocess_function = preprocess_data_record
        
        # word sense disambiguation task
        elif self.task_name in ["wic"]:
            def preprocess_function_wic(examples):
                sentences = []
                # concat all three input parts with [SEP] token
                for parts in zip(*[examples[c] for c in ["sentence1", "sentence2", "word"]]):
                    sentences.append(self.tokenizer.sep_token.join(parts))
                encoded = self.tokenizer(
                    sentences,
                    truncation=True,
                    padding="max_length",
                )
                encoded.update({"label": examples["label"]})
                return encoded
            preprocess_function = preprocess_function_wic
        
        # coreference resolution task
        elif self.task_name in ["wsc"]:
            def preprocess_function_wsc(examples):

                sentences = examples["text"]
                pronoun = examples["span1_text"]
                target = examples["span2_text"]
                span1_index = examples["span1_index"]
                span2_index = examples["span2_index"]
                labels = examples['label']

                # Encode the sentences
                encoded = self.tokenizer(
                    sentences,
                    truncation=True,
                    padding='max_length',
                    return_tensors="pt"
                )

                # Add token type ids for pronoun and target
                token_type_ids = torch.zeros_like(encoded["input_ids"])
                for i, (p_index, t_index) in enumerate(zip(span1_index, span2_index)):
                    token_type_ids[i, p_index] = 1
                    token_type_ids[i, t_index] = 1

                # Add labels to the processed examples
                processed_examples = {
                    'input_ids': encoded['input_ids'],
                    'attention_mask': encoded['attention_mask'],
                    'token_type_ids': token_type_ids,
                    'label': torch.tensor(labels)
                }

                return processed_examples

            preprocess_function = preprocess_function_wsc
        return preprocess_function

    def get_metric(self):
        if self.task_name in self.GLUE_TASKS:
            actual_task = "mnli" if self.task_name == "mnli_mismatched" or self.task_name == "mnli_matched" else self.task_name
            metric = load_metric('glue', actual_task)
        elif self.task_name in self.SUPERGLUE_TASKS:
            metric = load_metric('super_glue', self.task_name)
        return metric
    
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = None
        
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding="max_length",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
# example usage
# GLUE_TASKS = ["mrpc", "stsb", "rte", "wnli", "qqp", "mnli_mismatched", "mnli_matched", "qnli", "cola", "sst2" ]
# SUPERGLUE_TASKS = ["boolq", "cb", "copa", "multirc", "record", "wic",  "wsc"]
# data = GlueDataloader("copa")
# samples = data.get_samples(10)
# train, val = data.get_train_val_split()
# ipdb.set_trace()