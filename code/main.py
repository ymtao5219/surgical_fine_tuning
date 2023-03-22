import numpy as np
import ipdb
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from neuron_extractor import *
from neuron_analyzer import *

# Load dataset
dataset_name = "ptb_text_only"
dataset = load_dataset(dataset_name, split="test").select(range(500))

# Load the pretrained model
model_checkpoint_pretrained = "bert-base-multilingual-cased"
model_pretrained = AutoModel.from_pretrained(model_checkpoint_pretrained)
tokenizer_pretrained = AutoTokenizer.from_pretrained(model_checkpoint_pretrained)
cls_emb_pretrained = NeuronExtractor(model_pretrained, tokenizer_pretrained).extract_cls(dataset["sentence"])

# Load the finetuned model
model_checkpoint_finetuned = "QCRI/bert-base-multilingual-cased-pos-english"
model_finetuned = AutoModel.from_pretrained(model_checkpoint_finetuned)
tokenizer_finetuned = AutoTokenizer.from_pretrained(model_checkpoint_finetuned)
cls_emb_finetuned = NeuronExtractor(model_finetuned, tokenizer_finetuned).extract_cls(dataset["sentence"])


# Analysis 
NeuronAnalyzer(cls_emb_pretrained, cls_emb_finetuned).plot_neuron_histograms(top_k=10)