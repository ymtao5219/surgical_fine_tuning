import random
from typing import List
from datasets import load_dataset

'''
A class to sample sentences from GLUE Benchmark datasets.
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
        # elif dataset_name in ["qqp"]:
        #     self.sentences.extend([data["question1"], data["question2"]])
        # elif dataset_name in ["mnli_mismatched", "mnli_matched" "ax"]:
        #     self.sentences.extend([data["premise"], data["hypothesis"]])
        # elif dataset_name in ["qnli"]:
        #     self.sentences.extend([data["question"], data["sentence"]])

    def sample_sentences(self, seed, num_sentences) -> List[str]:
        if len(self.sentences) >= num_sentences:
            random.seed(seed)
            return random.sample(self.sentences, num_sentences)
        else:
            print("Not enough sentences available in the datasets.")
            return []

# Example usage:
# sampler = GLUESampler(seed=1314, num_sentences=100)
# sampled_sentences = sampler.sample_sentences()

# for i, sentence in enumerate(sampled_sentences):
#     print(f"Sample {i + 1}: {sentence}")
