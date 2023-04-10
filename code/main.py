import numpy as np
import ipdb
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
from neuron_extractor import *
from neuron_analyzer import *
from utils import *
from collections import defaultdict
from datasets import load_dataset

from scipy.stats import ttest_ind, ks_2samp

###########################################################################
# load data 
###########################################################################

dataset_name = "sst2"
data = load_dataset("glue", dataset_name)

validation_data = data["validation"]
label_sentences = {}
# Iterate over the validation data
for row in validation_data:
    label = row['label']
    sentence = row['sentence']

    # Add the sentence to the list corresponding to its label
    if label not in label_sentences:
        label_sentences[label] = []

    if len(label_sentences[label]) < 100:
        label_sentences[label].append(sentence)

###########################################################################
# model to be investigated 
###########################################################################
model_ckpts = {"before_finetuning": "bert-base-uncased", "after_finetuning": "nateraw/bert-base-uncased-emotion"}

###########################################################################
# experiment 1: for shuffled/unshuffled sentences, compare their activations 
# on top/bottom neurons using histogram, both before and after finetuning
###########################################################################
for model_type, ckpt in model_ckpts.items():
    model = AutoModel.from_pretrained(ckpt)
    tokenizer= AutoTokenizer.from_pretrained(ckpt)
    config = AutoConfig.from_pretrained(ckpt)

    # extract cls embedding for pairs of sentences
    cls_emb_pos_sentiment = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[1], layer_num=-1)
    cls_emb_neg_sentiment = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[0], layer_num=-1)
    # plot histogram for top/bottom neurons
    NeuronAnalyzer(cls_emb_pos_sentiment, cls_emb_neg_sentiment).plot_neuron_histograms(metric=ks_2samp, k = 10, model_type=model_type,
                                                                                       labels=["positive", "negative"], neuron_type="top")

    NeuronAnalyzer(cls_emb_pos_sentiment, cls_emb_neg_sentiment).plot_neuron_histograms(metric=ks_2samp, k = 10, model_type=model_type,
                                                                                       labels=["positive", "negative"], neuron_type="bottom")
   
###########################################################################
# experiment 2: for sentences labeled with positive/negative sentiment, plot
# a binary matrix, where each row corresponds to a layer, and each column corresponds 
# to a neuron. the value of each entry is 1 if the neuron is activated
###########################################################################
for model_type, ckpt in model_ckpts.items():
    model = AutoModel.from_pretrained(ckpt)
    tokenizer= AutoTokenizer.from_pretrained(ckpt)
    config = AutoConfig.from_pretrained(ckpt)
    num_layers = config.num_hidden_layers
    res = []
    num_of_activated_neurons = []
    for i in range(1, num_layers+1):
        # extract cls embedding for pairs of sentences
        cls_emb_with = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[1], layer_num=i)
        cls_emb_without = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[0], layer_num=i)
        # extract top activated neurons
        binary_vector = NeuronAnalyzer(cls_emb_with, cls_emb_without).rank_neuron(metric=ks_2samp, neuron_type="all")
        num_of_activated_neurons.append(np.sum(binary_vector))
        res.append(binary_vector)
    plot_heatmap(res, file_name=f"heatmap_{model_type}_{dataset_name}")
    print("the number of activated neurons at layer ", num_of_activated_neurons)
# stacked_arrays = np.vstack(res)
# save_numpy_array_to_file(stacked_arrays, f"./data/binary_matrix.npy")