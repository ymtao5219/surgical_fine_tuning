import numpy as np
import ipdb
import os
from transformers import AutoModel, AutoTokenizer
from neuron_extractor import *
from neuron_analyzer import *
from utils import *
from collections import defaultdict

# Load the pretrained model

# model_checkpoint = "QCRI/bert-base-multilingual-cased-pos-english"
model_checkpoint = "bert-base-multilingual-cased"
model = AutoModel.from_pretrained(model_checkpoint)
tokenizer= AutoTokenizer.from_pretrained(model_checkpoint)

attr2neuron =  defaultdict(list)

# read json 
for f in os.listdir('data'):
    data = read_json('data/' + f)
    attr = f.split(".")[0]
    sentences_with = data[f"with_{attr}"]
    sentences_without = data[f"without_{attr}"]
    
    # extract cls embedding for pairs of sentences
    cls_emb_with = NeuronExtractor(model, tokenizer).extract_cls(sentences_with)
    cls_emb_without = NeuronExtractor(model, tokenizer).extract_cls(sentences_without)
    
    # extract top activated neurons
    neuron_indices_top = NeuronAnalyzer(cls_emb_with, cls_emb_without).rank_neuron(neuron_type = "top")
    neuron_indices_bottom = NeuronAnalyzer(cls_emb_with, cls_emb_without).rank_neuron(neuron_type = "bottom")
    
    attr2neuron["top"] = neuron_indices_top
    attr2neuron["bottom"] = neuron_indices_bottom
    
    # construct graph
    NeuronAnalyzer(cls_emb_with, cls_emb_without).plot_neuron_histograms(neuron_type = "top")
    NeuronAnalyzer(cls_emb_with, cls_emb_without).plot_neuron_histograms(neuron_type = "bottom")
    
print(attr2neuron)

    