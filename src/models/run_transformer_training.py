import os
import sys
import time

sys.path.append(os.getcwd())
from src.models.pytorch_model import PyTorchModel
from src.helpers.wandb_helper import initialize_wandb
from src.helpers.seed_helper import initialize_gpu_seed
from src.models.config import read_arguments_train
from src.helpers.logging_helper import setup_logging
from src.helpers.path_helper import *


setup_logging()


def main(args):
    initialize_gpu_seed(args.model_seed)

    if args.wandb:
        initialize_wandb(args)

    model = PyTorchModel(args)
    model.train()


if __name__ == '__main__':
    args = read_arguments_train()
    start_time = time.time()
    main(args)
    final_time = time.time() - start_time
    hours, rem = divmod(final_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    with open(experiment_file_path(args.experiment_name, 'training_time.txt'), 'w') as f:
        f.write("Training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))