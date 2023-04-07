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

# Load the pretrained model

model_checkpoint = "bert-base-uncased"
# model_checkpoint = "QCRI/bert-base-multilingual-cased-pos-english"
# model_checkpoint = "bert-base-multilingual-cased"
model = AutoModel.from_pretrained(model_checkpoint)
tokenizer= AutoTokenizer.from_pretrained(model_checkpoint)
config = AutoConfig.from_pretrained(model_checkpoint)
num_layers = config.num_hidden_layers

# the following for-loop is for loading probe datasets that are stored in a json file
# attr2neuron =  defaultdict(list)
# for f in os.listdir('./data'):
#     data = read_json('./data/' + f)
#     attr = f.split(".")[0]
#     sentences_with = data[f"with_{attr}"]
#     sentences_without = data[f"without_{attr}"]
#     # extract cls embedding for pairs of sentences
#     cls_emb_with = NeuronExtractor(model, tokenizer).extract_layer_embedding(sentences_with)
#     cls_emb_without = NeuronExtractor(model, tokenizer).extract_layer_embedding(sentences_without)
#     # extract top activated neurons
#     neuron_indices = NeuronAnalyzer(cls_emb_with, cls_emb_without).rank_neuron(neuron_type="top", k=10, method="ks_stat")
#     # construct graph
#     attr2neuron[attr] = neuron_indices
#     # ipdb.set_trace()

data = load_dataset("glue", "sst2")

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

res = []

for i in range(1, num_layers+1):
    # extract cls embedding for pairs of sentences
    cls_emb_with = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[1], layer_num=i)
    cls_emb_without = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[0], layer_num=i)
    # extract top activated neurons
    binary_vector = NeuronAnalyzer(cls_emb_with, cls_emb_without).rank_neuron(metric=ks_2samp, neuron_type="all")
    print(f"the number of activated neurons at layer {i}", np.sum(binary_vector))
    res.append(binary_vector)

# stacked_arrays = np.vstack(res)
# save_numpy_array_to_file(stacked_arrays, f"./data/binary_matrix.npy")

plot_heatmap(res)


    