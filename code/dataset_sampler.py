import random
from typing import List
from datasets import load_dataset
import csv 
import random

'''
Save and randomly shuffle all sentences on the TEST sets from GLUE datasets to a CSV file.
'''

class GLUESampler:
    def __init__(self):
        #  ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'qnli', 'rte', 'wnli', 'ax',  'mnli_mismatched', 'mnli_matched', 'mnli']
        self.datasets = ["cola", "sst2", "mrpc", "qqp", "stsb", "qnli", "rte", "wnli",  "ax", 'mnli_mismatched', 'mnli_matched']
        self.sentences = []

        # 
        self.load_glue_datasets()

    def load_glue_datasets(self):
        for dataset_name in self.datasets:
            try:
                dataset = load_dataset("glue", dataset_name, split="test")
            except Exception as e:
                print(f"Failed to load dataset '{dataset_name}'. Reason: {e}")
                continue

            for example in dataset:
                self.process_data(dataset_name, example)

    def process_data(self, dataset_name: str, data: dict):

        if dataset_name in ["cola", "sst2"]:
            self.sentences.append(data["sentence"])
        elif dataset_name in ["mrpc", "stsb", "rte", "wnli"]:
            self.sentences.extend([data["sentence1"], data["sentence2"]])
        elif dataset_name in ["qqp"]:
            self.sentences.extend([data["question1"], data["question2"]])
        elif dataset_name in ["mnli_mismatched", "mnli_matched" "ax"]:
            self.sentences.extend([data["premise"], data["hypothesis"]])
        elif dataset_name in ["qnli"]:
            self.sentences.extend([data["question"], data["sentence"]])

    def sample_sentences(self, seed, num_sentences) -> List[str]:
        if len(self.sentences) >= num_sentences:
            random.seed(seed)
            return random.sample(self.sentences, num_sentences)
        else:
            print("Not enough sentences available in the datasets.")
            return []

def shuffle_csv_rows(file_name):
    with open(file_name, 'r') as csv_file:
        # Read the CSV file
        reader = csv.reader(csv_file)
        
        # Store all rows in a list
        rows = [row for row in reader]
        
    # Shuffle the rows
    random.shuffle(rows)
    
    # Write the shuffled rows to a new file
    output_file = 'shuffled_' + file_name
    with open(output_file, 'w', newline='') as shuffled_csv:
        writer = csv.writer(shuffled_csv)
        
        # Write the shuffled rows
        writer.writerows(rows)
        
    print(f'Shuffled CSV saved as {output_file}')

def list_to_csv(lst, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in lst:
            writer.writerow([row])

# save the sampled sentences to a csv file
# res = GLUESampler().sentences
# list_to_csv(res, "glue_sentences.csv")
# shuffle_csv_rows("glue_sentences.csv")
