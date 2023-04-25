from datasets import load_dataset
import random

class GlueDataloader:
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    SUPERGLUE_TASKS = ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"]

    def __init__(self, task_name, split="train", num_samples=None):
        self.task_name = task_name.lower()
        self.split = split
        self.num_samples = num_samples
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if self.task_name in self.GLUE_TASKS:
            dataset = load_dataset("glue", self.task_name)
        elif self.task_name in self.SUPERGLUE_TASKS:
            dataset = load_dataset("super_glue", self.task_name)
        else:
            raise ValueError("Invalid task name. Please choose from GLUE or SuperGLUE tasks.")

        if self.num_samples:
            dataset = self._sample_sentences(dataset, self.split, self.num_samples)
        
        return dataset

    def _sample_sentences(self, dataset, split, num_samples):
        sampled_data = dataset[split].select(range(min(len(dataset[split]), num_samples)))
        return sampled_data

    def get_data(self, split=None):
        if split:
            return self.dataset[split]
        else:
            return self.dataset
