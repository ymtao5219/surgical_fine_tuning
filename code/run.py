import argparse
from utils import *
from fim_calculator import *
from fine_tuner import *
from data_loader import *

import ipdb

def main(args):
    # todo: test for different models
    
    GLUE_TASKS = ["mrpc", "stsb", "rte", "wnli", "qqp", "mnli_mismatched", "mnli_matched", "qnli", "cola", "sst2" ]
    SUPERGLUE_TASKS = ["wic", "cb", "boolq", "copa", "multirc", "record", "wsc"]
    ipdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get average density per layer for BERT models")
    
    parser.add_argument("--model", type=str, default="bert-base-cased", help="model name")
    parser.add_argument("--model_type", type=str, default="before_fintuning", help="Model type")
    
    # dataset parameters
    parser.add_argument("--task_name", type=str, default="sst2", help="GLUE/SuperGlue task name")
    
    # sampling parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_of_sentences", type=int, default=100, help="Num of sentences to probe")
    
    main(parser.parse_args())
    # ipdb.set_trace()