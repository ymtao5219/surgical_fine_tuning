import torch
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    GlueDataset,
    GlueDataTrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from collections import defaultdict
import ipdb

# ["cola", "sst2", "mrpc", "qqp", "stsb", "qnli", "rte", "wnli",  "ax", 'mnli_mismatched', 'mnli_matched']

class FIMCalculator:
    def __init__(self, model_name, glue_task_name, glue_data_dir):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        data_args = GlueDataTrainingArguments(task_name=glue_task_name, data_dir=glue_data_dir)
        self.dataset = GlueDataset(data_args, tokenizer=self.tokenizer)

    def compute_fim_diagonal(self, batch_size=1):
        self.model.train()
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        fim_diagonals = []

        for batch in data_loader:
            inputs = {key: val.to(self.device) for key, val in batch.items()}
            self.model.zero_grad()
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    fim_diagonal = (param.grad ** 2).detach().cpu().numpy()
                    fim_diagonals.append((name, fim_diagonal))

        return fim_diagonals

    def organize_fim_per_layer(self, fim_diagonals):
        fim_per_layer = defaultdict(list)

        for name, fim_diagonal in fim_diagonals:
            layer_name = name.split('.')[0]
            fim_per_layer[layer_name].append(fim_diagonal)

        return fim_per_layer


model_name = 'bert-base-uncased'
glue_task_name = 'cola'
glue_data_dir = './glue_data/CoLA'
ipdb.set_trace()

fim_calculator = FIMCalculator(model_name, glue_task_name, glue_data_dir)
fim_diagonals = fim_calculator.compute_fim_diagonal(batch_size=1)
fim_per_layer = fim_calculator.organize_fim_per_layer(fim_diagonals)
