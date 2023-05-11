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
        columns_to_remove = [col for col in sampled_data.column_names if col != 'label']
        sampled_data = sampled_data.map(preprocess_function, batched=False, remove_columns=columns_to_remove)
        
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
            # ipdb.set_trace()
            columns_to_remove = [col for col in train_dataset.column_names if col != 'label']
            train_dataset = train_dataset.map(preprocess_function, batched=False, remove_columns=columns_to_remove)
            val_dataset = val_dataset.map(preprocess_function, batched=False, remove_columns=columns_to_remove)

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
            columns_to_remove = [col for col in train_dataset.column_names if col != 'label']
            train_dataset = selected_dataset.map(preprocess_function, batched=False, remove_columns=columns_to_remove)
            val_dataset = val_dataset.map(preprocess_function, batched=False, remove_columns=columns_to_remove)

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

        # NLI task 
        elif self.task_name in ["cb"]:
            def preprocess_function_cb(examples):
                encoded = self.tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')
                encoded['label'] = examples['label']
                return encoded
            preprocess_function = preprocess_function_cb

        # qa task 
        elif self.task_name in ["copa"]:
            def preprocess_function_copa(examples):
                # Check if we are working with a single example or a batch
                is_single_example = isinstance(examples["premise"], str)

                # Handle single examples
                if is_single_example:
                    question_header = examples["question"]
                    sentences = [
                        [f"{examples['premise']} What was the {question_header} of this?", f"{examples['choice1']}"],
                        [f"{examples['premise']} What was the {question_header} of this?", f"{examples['choice2']}"]
                    ]
                    labels = examples['label']

                # Handle batches
                else:
                    question_headers = examples["question"]
                    sentences = []
                    labels = []
                    for premise, question, choice1, choice2, label in zip(examples['premise'], question_headers, examples['choice1'], examples['choice2'], examples['label']):
                        sentences.append([f"{premise} What was the {question} of this?", f"{choice1}"])
                        sentences.append([f"{premise} What was the {question} of this?", f"{choice2}"])
                        labels.append(label)

                # Encode the sentences
                encoded = self.tokenizer(
                    sentences,
                    truncation=True,
                    padding='max_length',
                    return_tensors="pt"
                )

                # Combine the tokenized results
                input_ids = encoded['input_ids'].view(-1, 2, self.tokenizer.model_max_length)
                attention_mask = encoded['attention_mask'].view(-1, 2, self.tokenizer.model_max_length)
                token_type_ids = encoded['token_type_ids'].view(-1, 2, self.tokenizer.model_max_length)
                
                # Add labels to the processed examples
                processed_examples = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'label': torch.tensor(labels)
                }

                return processed_examples
            preprocess_function = preprocess_function_copa
        
        # qa task 
        elif self.task_name in ["multirc"]:
            def preprocess_data_multirc(example):
                # Get passage, questions, answers, and labels from the example
                passage = example['paragraph']
                questions = example['question']
                answers = example['answer']
                labels = example['label']

                # Tokenize the pairs of sentences
                encoded = self.tokenizer(
                    passage,
                    questions,
                    answers,
                    truncation=True,
                    padding='max_length'
                )
                # ipdb.set_trace()
                # Use the provided labels
                encoded['label'] = labels

                return encoded
            preprocess_function = preprocess_data_multirc
        
        # qa task 
        elif self.task_name in ["record"]:
            '''
              ReCoRD contains a passage, query containing a '@placeholder' string, and a set
                of entities that are the possible values of the placeholder. Each train and
                validation example will have a list of answers, any of which would be
                considered correct.

                For example, a typical example from ReCoRD might look like
                {
                    'passsage': 'This is the passage.',
                    'query': 'A @placeholder is a bird.',
                    'entities': ['penguin', 'potato', 'pigeon'],
                    'answers': ['penguin', 'pigeon'],
                }
            '''
            def preprocess_data_record(examples):

                passage_text = examples['passage'][0]
                query = examples['query'][0] # .replace('@placeholder', '[MASK]')
                entities = examples['entities']
                correct_entities = examples['answers']

                label = False
                if entities:
                    for entity in entities: 
                        if entity in correct_entities:
                            label = True
                # ipdb.set_trace()
                # entity_spans = example['entity_spans']
                # answer_dict = {}
                # for i in range(len(entity_spans['text'])):
                #     key = (entity_spans['start'][i], entity_spans['end'][i])
                #     value = entity_spans['text'][i]
                #     answer_dict[key] = value

                input_text = f"{query} [SEP] {passage_text}"
                input_encoding = self.tokenizer(
                    input_text,
                    truncation=True,
                    padding='max_length'
                )

                input_ids = input_encoding['input_ids']
                attention_mask = input_encoding['attention_mask']
                token_type_ids = input_encoding['token_type_ids']
                
                # ipdb.set_trace()
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'label': torch.tensor(label, dtype=torch.long),
                }
            preprocess_function = preprocess_data_record
        
        # word sense disambiguation task
        elif self.task_name in ["wic"]:
            preprocess_function = lambda examples: self.tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')
        
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
    
# example usage
# GLUE_TASKS = ["mrpc", "stsb", "rte", "wnli", "qqp", "mnli_mismatched", "mnli_matched", "qnli", "cola", "sst2" ]
# SUPERGLUE_TASKS = ["boolq", "cb", "copa", "multirc", "record", "wic",  "wsc"]
# data = GlueDataloader("record")
# samples = data.get_samples(10)
# train, val = data.get_train_val_split()
# ipdb.set_trace()