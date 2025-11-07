import os
import sys
import argparse
import logging
import time

sys.path.append(os.getcwd())
from src.matching.matcher_subclasses import SynCompanyMatcher, WDCMatcher
from src.models.pytorch_model import PyTorchModel
from src.models.config import update_args_with_config
from src.data.dataset import ExperimentDataset
from src.helpers.seed_helper import initialize_gpu_seed
from src.helpers.logging_helper import setup_logging
from src.helpers.path_helper import *
from src.helpers.matcher_helper import matchers_dict
from src.CLER.utils import *

setup_logging()



def read_arguments_TransClean():
    parser = argparse.ArgumentParser(description='Test model with following arguments')
    # Arguments from read_arguments_test() of src.models.config
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=False, default=None)
    parser.add_argument('--use_validation_set', action='store_true')
    # Arguments for the matcher:
    parser.add_argument('--matcher', type=str, required=True, choices=list(matchers_dict.keys()))
    parser.add_argument('--num_ds', type=int, required=False, default=5)
    parser.add_argument('--manual_check', action='store_true') # Whether to manually check the edges selected for fine-tuning or weakly label them all as 0s
    parser.add_argument('--remove_true_positives', action='store_true') # Whether to remove the true positives from the finetuning edges
    parser.add_argument('--eval_positive_pairwise_preds', action='store_true') # Whether to evaluate all the positive pairwise predictions with the further finetuned model or only the deleted edges
    # Argument for the threshold used in the matching
    parser.add_argument('--threshold', type=float, required=False, default=0.999)
    # Arguments for the finetuning
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_config', action='store_true')
    parser.add_argument('--finetuning_epochs', type=int, required=False, default=3) # Number of epochs to finetune the model
    parser.add_argument('--finetuning_iterations', type=int, required=False, default=5) # Number of times to finetune the model

    args = parser.parse_args()
    args = update_args_with_config(args.experiment_name, args)

    for argument in vars(args):
        logging.info("argument: {} =\t{}".format(str(argument).ljust(20), getattr(args, argument)))

    return args


def main(args):
    initialize_gpu_seed(args.seed)

    if 'CLER' in args.experiment_name:
        model = load_CLER_matcher(args.experiment_name)
        model.args = args
        model.dataset = ExperimentDataset.create_instance(args.dataset_name, args.model_name, seed=args.seed)
        
    else:
        checkpoint_suffix = '__epoch' + str(args.epoch)
        if args.epoch == 0:
            checkpoint_suffix += '__zeroshot'

        file_name = "".join([args.model_name, checkpoint_suffix, '.pt'])
        checkpoint_path = experiment_file_path(args.experiment_name, file_name)

        model = PyTorchModel.load_from_checkpoint(args, checkpoint_path)

    matcher = matchers_dict[args.matcher](model=model, num_ds=args.num_ds)

    # Run the matching w/ finetuning
    args.use_validation_set = False
    matcher.run_TransClean(args)

    return model

if __name__ == '__main__':
    args = read_arguments_TransClean()

    for labeling_budget in [1000]:
        start_time = time.time()
        args.labeling_budget = labeling_budget
        model = main(args)
        final_time = time.time() - start_time
        hours, rem = divmod(final_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Cleanup time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        with open(os.path.join(dataset_results_folder_path__with_subfolders(subfolder_list=[model.dataset.name, model.args.experiment_name, 'labeling_budget_{}'.format(args.labeling_budget)]), 'cleanup_time.txt'), 'w') as f:
            f.write("Cleanup time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
