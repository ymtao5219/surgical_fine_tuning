import argparse
from utils import *
from fim_calculator import *
from fine_tuner import *
from data_loader import *

import ipdb

def main(args):
    
    GLUE_TASKS = [ "sst2", "cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "stsb", "wnli"]
    SUPERGLUE_TASKS = ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"]

    model_name = "bert-base-cased"
    glue_task_name = "mnli"
    num_sentences = 100

    fim_calculator = FIMCalculator(model_name, glue_task_name, num_sentences)
    fim_diag_by_layer = fim_calculator.compute_fim(batch_size=1, empirical=True, verbose=True, every_n=None)
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