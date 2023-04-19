import numpy as np
import ipdb
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
from neuron_extractor import *
from neuron_analyzer import *
from utils import *
from collections import defaultdict
from datasets import load_dataset
import random 

from dataset_sampler import *

from scipy.stats import ttest_ind, ks_2samp

# ###########################################################################
# # experiment 1: for shuffled/unshuffled sentences, compare their activations 
# # on top/bottom neurons using histogram, both before and after finetuning
# ###########################################################################
# model_ckpts = {"before_finetuning": "bert-base-uncased", "after_finetuning": "nateraw/bert-base-uncased-emotion"}
# dataset_name = "sst2"
# data = load_dataset("glue", dataset_name)

# validation_data = data["validation"]
# label_sentences = {}
# # Iterate over the validation data
# for row in validation_data:
#     label = row['label']
#     sentence = row['sentence']

#     # Add the sentence to the list corresponding to its label
#     if label not in label_sentences:
#         label_sentences[label] = []

#     if len(label_sentences[label]) < 100:
#         label_sentences[label].append(sentence)
        
# for model_type, ckpt in model_ckpts.items():
#     model = AutoModel.from_pretrained(ckpt)
#     tokenizer= AutoTokenizer.from_pretrained(ckpt)
#     config = AutoConfig.from_pretrained(ckpt)

#     # extract cls embedding for pairs of sentences
#     cls_emb_pos_sentiment = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[1], layer_num=-1)
#     cls_emb_neg_sentiment = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[0], layer_num=-1)
#     # plot histogram for top/bottom neurons
#     NeuronAnalyzer(cls_emb_pos_sentiment, cls_emb_neg_sentiment).plot_neuron_histograms(metric=ks_2samp, k = 10, model_type=model_type,
#                                                                                        labels=["positive", "negative"], neuron_type="top")

#     NeuronAnalyzer(cls_emb_pos_sentiment, cls_emb_neg_sentiment).plot_neuron_histograms(metric=ks_2samp, k = 10, model_type=model_type,
#                                                                                        labels=["positive", "negative"], neuron_type="bottom")
   
# ###########################################################################
# # experiment 2: for sentences labeled with positive/negative sentiment, plot
# # a binary matrix, where each row corresponds to a layer, and each column corresponds 
# # to a neuron. the value of each entry is 1 if the neuron is activated
# ###########################################################################
# for model_type, ckpt in model_ckpts.items():
#     model = AutoModel.from_pretrained(ckpt)
#     tokenizer= AutoTokenizer.from_pretrained(ckpt)
#     config = AutoConfig.from_pretrained(ckpt)
#     num_layers = config.num_hidden_layers
#     res = []
#     num_of_activated_neurons = []
#     for i in range(1, num_layers+1):
#         # extract cls embedding for pairs of sentences
#         cls_emb_with = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[1], layer_num=i)
#         cls_emb_without = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[0], layer_num=i)
#         # extract top activated neurons
#         binary_vector = NeuronAnalyzer(cls_emb_with, cls_emb_without).rank_neuron(metric=ks_2samp, neuron_type="all")
#         num_of_activated_neurons.append(np.sum(binary_vector))
#         res.append(binary_vector)
#     plot_heatmap(res, file_name=f"heatmap_{model_type}_{dataset_name}")
#     print("the number of activated neurons at layer ", num_of_activated_neurons)
        
###########################################################################
# experiment 3: same experiment, but for cola dataset, where the labels are
# grammatically correct/incorrect
###########################################################################
# dataset_name = "cola"
# data = load_dataset("glue", dataset_name)

# validation_data = data["validation"]
# label_sentences = {}
# # Iterate over the validation data
# for row in validation_data:
#     label = row['label']
#     sentence = row['sentence']

#     # Add the sentence to the list corresponding to its label
#     if label not in label_sentences:
#         label_sentences[label] = []

#     if len(label_sentences[label]) < 100:
#         label_sentences[label].append(sentence)

# # ipdb.set_trace()

# model_ckpts = {"before_finetuning": "bert-base-uncased", "after_finetuning": "textattack/bert-base-uncased-CoLA"}

# for model_type, ckpt in model_ckpts.items():
#     model = AutoModel.from_pretrained(ckpt)
#     tokenizer= AutoTokenizer.from_pretrained(ckpt)
#     config = AutoConfig.from_pretrained(ckpt)
#     num_layers = config.num_hidden_layers
#     res = []
#     num_of_activated_neurons = []
#     for i in range(1, num_layers+1):
#         # extract cls embedding for pairs of sentences
#         cls_emb_with = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[1], layer_num=i)
#         cls_emb_without = NeuronExtractor(model, tokenizer).extract_layer_embedding(label_sentences[0], layer_num=i)
#         # extract top activated neurons
#         binary_vector = NeuronAnalyzer(cls_emb_with, cls_emb_without).rank_neuron(metric=ks_2samp, neuron_type="all")
#         num_of_activated_neurons.append(np.sum(binary_vector))
#         res.append(binary_vector)
#     plot_heatmap(res, file_name=f"heatmap_{model_type}_{dataset_name}")
#     print("the number of activated neurons at layer ", num_of_activated_neurons)
    
    
###########################################################################
# experiment 4
###########################################################################
# model_ckpts = {"before_finetuning": "bert-base-uncased", "after_finetuning": "textattack/bert-base-uncased-SST-2"}
# dataset_name = "sst2"
# attr = "sentiment"

# model_ckpts = {"before_finetuning": "bert-base-uncased", "after_finetuning": "textattack/bert-base-uncased-CoLA"}
# dataset_name = "cola"
# attr = "entailment"

# data = load_dataset("glue", dataset_name)
# test_set = data['test']

# num_of_sentences = 100

# probe_set = random.sample(list(test_set['sentence']), num_of_sentences)
# ipdb.set_trace()

# # number of samples taken from larger dataset
# M = 3
# sampler = GLUESampler()

# for model_type, ckpt in model_ckpts.items():
#     model = AutoModel.from_pretrained(ckpt)
#     tokenizer= AutoTokenizer.from_pretrained(ckpt)
#     config = AutoConfig.from_pretrained(ckpt)
#     num_layers = config.num_hidden_layers
#     res = np.zeros((num_layers, 768))
    
#     for m in range(M): 
        
#         sampled_sentences = sampler.sample_sentences(m, num_of_sentences)
#         # print(sampled_sentences)
#         binary_vector_per_layer = []
#         for i in range(1, num_layers+1):
#             # extract cls embedding for pairs of sentences
#             cls_emb_with = NeuronExtractor(model, tokenizer).extract_layer_embedding(probe_set, layer_num=i)
#             cls_emb_without = NeuronExtractor(model, tokenizer).extract_layer_embedding(sampled_sentences, layer_num=i)
#             # extract top activated neurons
#             binary_vector = NeuronAnalyzer(cls_emb_with, cls_emb_without).rank_neuron(metric=ks_2samp, neuron_type="all")
#             binary_vector_per_layer.append(binary_vector)
#         res += np.vstack(binary_vector_per_layer)
#     res = res / M
#     plot_heatmap(res, file_name=f"average_density_{model_type}_{dataset_name}")
#     save_numpy_array_to_file(res, f"average_density_{attr}_{model_type}_{dataset_name}_M{M}.npy")

# # attr = "sentiment"
# # M = 3 
# density1 = load_numpy_array_from_file(f"average_density_{attr}_before_finetuning_{dataset_name}_M{M}.npy")
# density2 = load_numpy_array_from_file(f"average_density_{attr}_after_finetuning_{dataset_name}_M{M}.npy")
# print(np.average(density1, axis=1))
# print(np.average(density2, axis=1))

# print(np.average(density1))
# print(np.average(density2))
# # threshold1 = 0.9

# # mask1 = density1 > threshold1
# # density1[mask1] = 1
# # density1[~mask1] = 0

# # threshold2 = 0.9
# # mask2 = density2 > threshold2
# # density2[mask2] = 1
# # density2[~mask2] = 0

# plot_heatmap(density1, file_name=f"average_density_{attr}_before_finetuning_sst2_M{M}_")
# plot_heatmap(density2, file_name=f"average_density_{attr}_after_finetuning_sst2_M{M}_")

###########################################################################
# experiment 5: with bonferroni correction
###########################################################################

num_of_sentences = 100

# probe dataset 1
# model_ckpts = {"before_finetuning": "bert-base-uncased", "after_finetuning": "textattack/bert-base-uncased-SST-2"}
# dataset_name = "sst2"
# attr = "sentiment"
# data = load_dataset("glue", dataset_name)
# test_set = data['test']
# probe_set = random.sample(list(test_set['sentence']), num_of_sentences)

# probe dataset 2
# model_ckpts = {"before_finetuning": "bert-base-uncased", "after_finetuning": "textattack/bert-base-uncased-CoLA"}
# dataset_name = "cola"
# attr = "entailment"
# data = load_dataset("glue", dataset_name)
# test_set = data['test']
# probe_set = random.sample(list(test_set['sentence']), num_of_sentences)

# probe dataset 3
model_ckpts = {"before_finetuning": "bert-base-uncased", "after_finetuning": "thanawan/bert-base-uncased-finetuned-humordetection"}
dataset_name = "Fraser/short-jokes"
attr = "humor"
data = load_dataset(dataset_name)
test_set = data['train']
probe_set = random.sample(list(test_set['text']), num_of_sentences)

# ipdb.set_trace()

# number of samples taken from larger dataset
M = 5
# samples taken from larger dataset
samples_glue = sample_sentences("glue_sentences.csv", M, num_of_sentences)

for model_type, ckpt in model_ckpts.items():
    model = AutoModel.from_pretrained(ckpt)
    tokenizer= AutoTokenizer.from_pretrained(ckpt)
    config = AutoConfig.from_pretrained(ckpt)
    num_layers = config.num_hidden_layers
    res = []
    
    for i in range(1, num_layers+1):
        # extract cls embedding for pairs of sentences
        cls_emb_with = NeuronExtractor(model, tokenizer).extract_layer_embedding(probe_set, layer_num=i)
        batched_tensors = []
        for m in range(M): 
            sampled_sentences = samples_glue[m]
            cls_emb_without = NeuronExtractor(model, tokenizer).extract_layer_embedding(sampled_sentences, layer_num=i)
            batched_tensors.append(cls_emb_without)
        
        activations2 = np.stack(batched_tensors, axis=0)
        binary_vector = NeuronAnalyzer(cls_emb_with, activations2).compute_test_statistic_bonferroni(metric=ks_2samp, alpha=0.01)
        res.append(binary_vector)

    res = np.stack(res, axis=0)
    res /= M
    plot_heatmap(res, file_name=f"average_density_{model_type}_{attr}_M{M}")
    save_numpy_array_to_file(res, f"average_density_{attr}_{model_type}_M{M}.npy")

# M = 3 
density1 = load_numpy_array_from_file(f"average_density_{attr}_before_finetuning_M{M}.npy")
density2 = load_numpy_array_from_file(f"average_density_{attr}_after_finetuning_M{M}.npy")
print(np.average(density1, axis=1))
print(np.average(density2, axis=1))

print(np.average(density1))
print(np.average(density2))
ipdb.set_trace()