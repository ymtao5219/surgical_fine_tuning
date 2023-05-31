import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import XLNetForSequenceClassification

import argparse
import re
import sys
import time
from typing import Dict, Optional
from torch import Tensor
import numpy as np 
import random 

from data_loader import *

import ipdb

# Set seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(1314)

class FIMCalculator:

    def __init__(self, model_name: str, tokenized_data):
        self.model_name = model_name
        self.tokenized_data = tokenized_data

        self.model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=len(tokenized_data.unique("label")))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_sentences = len(self.tokenized_data)
        self.tokenized_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    def compute_fim(self, batch_size=1, empirical=True, verbose=True, every_n=None):
        data_loader = DataLoader(self.tokenized_data, batch_size=batch_size)
        
        all_fims = self.fim_diag(self.model, 
                                 data_loader, 
                                 samples_no=self.num_sentences, 
                                 empirical=empirical, 
                                 device=self.device, 
                                 verbose=verbose, 
                                 every_n=every_n)
        
        fim_diag_by_layer = self.aggregate_fisher_information(all_fims)
        return fim_diag_by_layer

    @staticmethod
    def fim_diag(model: Module,
                data_loader: DataLoader,
                samples_no: int = None,
                empirical: bool = False,
                device: torch.device = None,
                verbose: bool = False,
                every_n: int = None) -> Dict[int, Dict[str, Tensor]]:
        model.eval()
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

            logits = model(data)[0]

            if empirical:
                outdx = target.unsqueeze(1).long()
            else:
                outdx = torch.argmax(logits, dim=1).unsqueeze(1).long().detach()
            samples = logits.gather(1, outdx)

            idx, batch_size = 0, data.size(0)
            while idx < batch_size and (samples_no is None or seen_no < samples_no):
                model.zero_grad()
                torch.autograd.backward(samples[idx], retain_graph=True)
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:  # Check for None gradient
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

            if layer_name == "transformer" and layer_name_parts[1] == "layer":
                layer_index_match = re.search(r'\d+', layer_name_parts[2])
                if layer_index_match is not None:
                    layer_index = layer_index_match.group()
                    layer_name = f"{layer_name}.layer_{layer_index}"

                if layer_name not in fim_diag_by_layer:
                    fim_diag_by_layer[layer_name] = 0.0

                fim_diag_by_layer[layer_name] += torch.norm(param_fim_diag, p='fro').item()

        return fim_diag_by_layer

    @staticmethod
    def bottom_k_layers(input_dict, k):
        sorted_items = sorted(input_dict.items(), key=lambda x: x[1])
        keys = [item[0] for item in sorted_items[:k]]
        return keys

if __name__ == "__main__":    
    # Example usage 
    # GLUE_TASKS = ["mrpc", "stsb", "rte", "wnli", "qqp", "mnli_mismatched", "mnli_matched", "qnli", "cola", "sst2"]
    # SUPERGLUE_TASKS = ["cb", "multirc", "wic", "wsc", "record", "copa", "boolq"]

    parser = argparse.ArgumentParser(description="Fisher score calculation")  
    parser.add_argument("--task_name", type=str, default="mrpc", help="Name of the task in GLUE/SuperGLUE")
 
    args = parser.parse_args()
    task_name = args.task_name

    model_name = "xlnet-base-cased"
    tokenized_data = GlueDataloader(task_name).get_samples(100)

    calc = FIMCalculator(model_name, tokenized_data)
    fim = calc.compute_fim(batch_size=1, empirical=True, verbose=True, every_n=None)

    # Select those with the lowest FIM layers to freeze
    layers_to_freeze = calc.bottom_k_layers(fim, k=12)
    ipdb.set_trace()

