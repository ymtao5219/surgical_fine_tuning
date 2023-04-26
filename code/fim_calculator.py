import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import re
import sys
import time
from typing import Dict, Optional
from torch import Tensor

import ipdb

class FIMCalculator:

    def __init__(self, model_name: str, glue_task_name: str, num_sentences: int):
        self.model_name = model_name
        self.glue_task_name = glue_task_name
        self.num_sentences = num_sentences

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.dataset = load_dataset("glue", glue_task_name)["validation"].shuffle(seed=42).select(range(num_sentences))

        self.tokenized_data = self.dataset.map(self.tokenize_function, batched=True, remove_columns=["sentence", "idx"])
        self.tokenized_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    def compute_fim(self, batch_size=1, empirical=True, verbose=True, every_n=None):
        data_loader = DataLoader(self.tokenized_data, batch_size=batch_size)
        all_fims = self.fim_diag(self.model, data_loader, samples_no=self.num_sentences, empirical=empirical, device=self.device, verbose=verbose, every_n=every_n)
        
        fim_diag_by_layer = self.aggregate_fisher_information(all_fims)
        return fim_diag_by_layer

    def tokenize_function(self, example):
        return self.tokenizer(example["sentence"], truncation=True, padding=True, return_tensors="pt")

    @staticmethod
    def fim_diag(model: Module,
                 data_loader: DataLoader,
                 samples_no: int = None,
                 empirical: bool = False,
                 device: torch.device = None,
                 verbose: bool = False,
                 every_n: int = None) -> Dict[int, Dict[str, Tensor]]:
        fim = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fim[name] = torch.zeros_like(param)

        seen_no = 0
        last = 0
        tic = time.time()

        all_fims = dict({})

        while samples_no is None or seen_no < samples_no:
            data_iterator = iter(data_loader)
            try:
                # data, target = next(data_iterator)
                batch = next(data_iterator)
                data, target = batch["input_ids"], batch["label"]
            except StopIteration:
                if samples_no is None:
                    break
                data_iterator = iter(data_loader)
                data, target = next(data_loader)

            if device is not None:
                data = data.to(device)
                if empirical:
                    target = target.to(device)

            logits = model(data).logits
            
            if empirical:
                outdx = target.unsqueeze(1)
            else:
                outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
            samples = logits.gather(1, outdx)

            idx, batch_size = 0, data.size(0)
            while idx < batch_size and (samples_no is None or seen_no < samples_no):
                model.zero_grad()
                torch.autograd.backward(samples[idx], retain_graph=True)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        fim[name] += (param.grad * param.grad)
                        fim[name].detach_()
                seen_no += 1
                idx += 1

                if verbose and seen_no % 100 == 0:
                    toc = time.time()
                    fps = float(seen_no - last) / (toc - tic)
                    tic, last = toc, seen_no
                    sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")

                if every_n and seen_no % every_n == 0:
                    all_fims[seen_no] = {n: f.clone().div_(seen_no).detach_()
                                        for (n, f) in fim.items()}

        if verbose:
            if seen_no > last:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
            sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n")

        for name, grad2 in fim.items():
            grad2 /= float(seen_no)

        all_fims[seen_no] = fim

        return all_fims

    @staticmethod
    def aggregate_fisher_information(all_fims):
        latest_fim_diag = all_fims[max(all_fims.keys())]
        fim_diag_by_layer = {}

        for param_name, param_fim_diag in latest_fim_diag.items():
            layer_name_parts = param_name.split('.')
            layer_name = layer_name_parts[0]

            if layer_name == "bert" and layer_name_parts[1] == "encoder":
                layer_index_match = re.search(r'\d+', layer_name_parts[3])
                if layer_index_match is not None:
                    layer_index = layer_index_match.group()
                    layer_name = f"{layer_name}.encoder.layer_{layer_index}"

            if layer_name not in fim_diag_by_layer:
                fim_diag_by_layer[layer_name] = 0.0

            fim_diag_by_layer[layer_name] += torch.norm(param_fim_diag, p='fro').item()

        return fim_diag_by_layer

# Example usage:
model_name = "bert-base-cased"
glue_task_name = "sst2"
num_sentences = 100

fim_calculator = FIMCalculator(model_name, glue_task_name, num_sentences)
fim_diag_by_layer = fim_calculator.compute_fim(batch_size=1, empirical=True, verbose=True, every_n=None)
ipdb.set_trace()