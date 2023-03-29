import numpy as np
import ipdb
import os
from transformers import AutoModel, AutoTokenizer
from neuron_extractor import *
from neuron_analyzer import *
from utils import *
from collections import defaultdict

# Load the pretrained model

model_checkpoint = "QCRI/bert-base-multilingual-cased-pos-english"
# model_checkpoint = "bert-base-multilingual-cased"
model = AutoModel.from_pretrained(model_checkpoint)
tokenizer= AutoTokenizer.from_pretrained(model_checkpoint)

attr2neuron =  defaultdict(list)

# read json 
for f in os.listdir('./data'):
    data = read_json('./data/' + f)
    attr = f.split(".")[0]
    sentences_with = data[f"with_{attr}"]
    sentences_without = data[f"without_{attr}"]
    # extract cls embedding for pairs of sentences
    cls_emb_with = NeuronExtractor(model, tokenizer).extract_cls(sentences_with)
    cls_emb_without = NeuronExtractor(model, tokenizer).extract_cls(sentences_without)
    # extract top activated neurons
    neuron_indices = NeuronAnalyzer(cls_emb_with, cls_emb_without).rank_neuron(top_k=10)
    # construct graph
    attr2neuron[attr] = neuron_indices
    
print(attr2neuron)

    