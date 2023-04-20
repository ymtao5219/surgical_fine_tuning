import numpy as np
import ipdb
from transformers import AutoModel, AutoTokenizer, AutoConfig

from datasets import load_dataset
import random 
import argparse

from neuron_extractor import *
from neuron_analyzer import *
from utils import *
from dataset_sampler import *

from scipy.stats import ttest_ind, ks_2samp

def main(args):
    if args.test_statistic == "ttest_ind":
        metric = ttest_ind
    if args.test_statistic == "ks_2samp":
        metric = ks_2samp
    
    # model ckpt 
    ckpt = args.model
    
    # probe set
    attr_name, probe_set = sample_probe_set(args.probe_set, args.num_of_sentences, args.seed)

    # large set of sentences
    negative_samples = sample_negative_sentences(args.negative_samples, args.num_of_negative_batches, args.num_of_sentences)
    
    model = AutoModel.from_pretrained(ckpt)
    tokenizer= AutoTokenizer.from_pretrained(ckpt)
    config = AutoConfig.from_pretrained(ckpt)
    num_layers = config.num_hidden_layers
    res = []
    
    for i in range(1, num_layers+1):
        # extract cls embedding for pairs of sentences
        cls_emb_with = NeuronExtractor(model, tokenizer).extract_layer_embedding(probe_set, layer_num=i)
        batched_tensors = []
        for m in range(args.num_of_negative_batches): 
            sample_m = negative_samples[m]
            cls_emb_without = NeuronExtractor(model, tokenizer).extract_layer_embedding(sample_m, layer_num=i)
            batched_tensors.append(cls_emb_without)
        
        activations2 = np.stack(batched_tensors, axis=0)
        binary_vector = NeuronAnalyzer(cls_emb_with, activations2).compute_test_statistic_bonferroni(metric=ks_2samp, alpha=0.01)
        res.append(binary_vector)

    res = np.stack(res, axis=0)
    res /= args.num_of_negative_batches
    average_density = np.average(res, axis=1)
    print(average_density.tolist())
    save_numpy_array_to_file(average_density, f"results/average_density_{attr_name}_{args.model_type}_N{args.num_of_sentences}_M{args.num_of_negative_batches}.npy")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get average density per layer for BERT models")
    
    parser.add_argument("--model", type=str, default="bert-base-multilingual-cased", help="model name")
    parser.add_argument("--model_type", type=str, default="before_fintuning", help="Model type")
    
    # data
    parser.add_argument("--probe_set", type=str, default="data/sst2.csv", help="Probe dataset name")
    parser.add_argument("--negative_samples", type=str, default="data/glue_sentences.csv", help="Dataset name for negative samples")
    
    # sampling parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_of_sentences", type=int, default=100, help="Num of sentences to probe")
    parser.add_argument("--num_of_negative_batches", type=int, default=10, help="Num of batches to take from negative samples")
    
    # test statistic
    # choice: ttest_ind, ks_2samp
    parser.add_argument("--test_statistic", type=str, default="ks_2samp", help="Test statistic to use for hypothesis testing")
    parser.add_argument("--alpha", type=float, default=0.01, help="Significance level for hypothesis testing")
    
    main(parser.parse_args())
    # ipdb.set_trace()