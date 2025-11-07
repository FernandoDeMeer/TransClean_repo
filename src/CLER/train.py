import os
import argparse
import json
import sys
import torch
import numpy as np
import pandas as pd 
import random
import warnings
import time

sys.path.append(os.getcwd())

from src.CLER.utils import *
from src.CLER.dataset import GTDatasetWithLabel
from src.CLER.runner import train

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__=="__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="wdc/shoes")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--add_token", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--CLlogdir", type=str, default="CL-sep-sup_0104")
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--total_budget", type=int, default=500)
    parser.add_argument("--warmup_budget", type=int, default=400)
    parser.add_argument("--active_budget", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--topK", type=int, default=5)
    parser.add_argument("--balance", type=bool, default=False)
    parser.add_argument("--valid_size", type=int, default=200)
    parser.add_argument("--blocker_type", type=str, default='sentbert') # sentbert/magellan
    parser.add_argument("--validation_with_pseudo", type=bool, default=False)
    parser.add_argument("--aug_type", type=str, default='random')
    
    hp = parser.parse_args()
    assert torch.cuda.is_available() == True 

    dataset = hp.dataset
    dataset_path_dict = {
        "synthetic_companies": "data/raw/synthetic_data/seed_0/companies",
        "wdc": "data/raw/wdc80_pair",
        "camera": "data/raw/camera",
        "monitor": "data/raw/monitor",
        'musicbrainz': 'data/raw/musicbrainz',
    }
    hp.path = dataset_path_dict[dataset]

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    train(hp)
    final_time = time.time() - start_time
    hours, rem = divmod(final_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Total training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    # Save the training time to the checkpoints folder
    with open(os.path.join('models',  hp.dataset + '_CLER_' + str(hp.total_budget) + '_seed_' + str(hp.run_id), 'elapsed_time_train.txt'), 'w') as f:
        f.write("Total training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print("Training Done!")