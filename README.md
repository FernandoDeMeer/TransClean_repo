# TransClean: Finding False Positives in Multi-Source Entity Matching under Real-World Conditions via Transitive Consistency

This repository includes the code base used in the paper "TransClean: Finding False Positives in Multi-Source Entity Matching under Real-World Conditions via Transitive Consistency".

## Authors

- Fernando De Meer Pardo*; University of Zurich, Zurich University of Applied Sciences, Winterthur, Switzerland; [fernando.demeerpardo@uzh.ch](mailto:fernando.demeerpardo@uzh.ch)
- Branka Hadji Misheva; Bern University of Applied Sciences, Bern, Switzerland; [branka.hadjimisheva@bfh.ch](mailto:branka.hadjimisheva@bfh.ch)
- Martin Braschler; Zurich University of Applied Sciences, Winterthur, Switzerland; [martin.braschler@zhaw.ch](mailto:martin.braschler@zhaw.ch)
- Kurt Stockinger; University of Zurich, Zurich University of Applied Sciences, Winterthur, Switzerland; [kurt.stockinger@zhaw.ch](mailto:kurt.stockinger@zhaw.ch)

*Corresponding author.

## Setup Your Environment

This readme assumes, you are running Python version `3.8.10`- though other recent versions of Python should work as well.

### Local Env
Use the following code snippets to set up your local environment:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

On Ubuntu, it might be necessary to install the dev version of Python, i.e. `sudo apt-get install python3-dev`.

## Repository Layout

- `data/` is structured as follows:
    - `data/raw/` contains raw record files and train/val/test splits for some datasets (those with pre-split samples)
    - `data/processed/` will contain the pre-processed files used during training & matching. Files ending in `__given_matches.csv` contain the original positive pairs included in each split and files ending `__all_matches.csv` include the randomly added negative pairs used in the training of some models.
    - `data/result/` will contain the result files of the training and TransClean scripts, indexed by the dataset and experiment names.
- `models/` will contain all of the model weights and prediction logs produced as a result of finetuning/training scripts.
- `scripts/` contains different scripts that do not implement any model and/or TransClean functionality.
- `src/`contains the implementations of TransClean and the models we combine it with (DistilBERT/transformers and CLER):
    - `src/CLER` contains CLER's implementation, minimally modified to run on our dataset (the [original implementation](https://github.com/wusw14/CLER/tree/master) works only in a 2-tables setting whereas we have 5 different data sources all in the same table).
    - `src/data` contains the implementation of the Preprocessor, BaseTokenizer and ExperimentDataset classes used to train DistilBERT.
    - `src/helpers` contains several files with helper functions for the different aspects of our methodology.
    - `src/matching` contains the Matcher class which implements TransClean. It also contains the Matcher subclasses which implement blocking and pre_cleanup functionalities.
    - `src/models` contains the PyTorchModel class which implements the training, validation, testing and additional finetuning of transformer models. We specifically choose DistilBERT in our experiments. 
# End-to-End Entity Matching Experiments

## Workflow Overview

For entity matching, the primary goal is to generate groups of records from different data sources that all represent the same entity. Our experiments run the following steps:

- Finetune a pairwise matching model on a set of manually labeled pairs.
- Evaluate with the finetuned model a set of candidate pairs (obtained through a blocking step).
- Run TransClean combined with the pairwise matching model and its set of predictions on the candidate pairs.
- Evaluate the performance of the pairwise matching model & TransClean and produce Figures.

## Start Running Code

Activate the installed Python environment either by running `source env/bin/activate` or executing the `activate_env.sh` shell script. 

## Download Synthetic Companies Dataset

Download the `seed_0` folder from this [Google Drive](https://drive.google.com/drive/folders/1KARFq_wKdmjL8d3JNElOiOkAu6O5Z7qY) and create a new folder called `synthetic_data` in `data/raw`. Place the `seed_0` folder in the newly created folder so you end up with the following folder path: `data/raw/synthetic_data/seed_0`.

### Finetune a transformer model

To train a new model, use the script `src/models/run_transformer_training.py` with the following arguments (see `src/models/config.py`, specifically `read_arguments_train()`):

| Argument             | Default Value                 | Description                                                                                                                                                                                                       |
|----------------------|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dataset_name`       | 'synthetic_companies_small' | Defines, which dataset is used for training. Use the keys in the DATASETS dictionary of `src/models/config.py` for reference.                                                                                     |
| `--model_name_or_path` | 'distilbert-base-uncased'     | The pre-trained large language model (LLM) to be used. See MODELS in `src/models/config.py` for reference. We use DistilBERT & CLER in our experiments                                                                            |
| `--model_name`         | 'distilbert'                  | same as `model_name_or_path`                                                                                                                                                                                              |
| `--experiment_name`    | None                          | Can be left empty, unless you specifically want to name the experiment. If left blank, an experiment name will be generated as a combination of the current datetime and the model used.                                                      |
| `--batch_size`        | 8                             | 8 is a good starting point. When increasing the batch size, more memory is required but the training is quicker. 
| `--max_seq_length`     | 128                           | Maximum toke sequence length the transformer model will be able to process, needs to match the model's input layer size. We use DistilBERT's 128 as a default.                                                                                              |
| `--do_lower_case`      | True                          | Whether or not to lowercase the input when tokenizing.                                                                                                 |
| `--nonmatch_ratio`     | 5                             | Ratio of non-matches to randomly add to the finetuning samples.                                                                                        |
| `--seed`               | 43                            | This seed is used in case the RANDOM SplitMethod is selected (which will randomly split a set of positive matches across different split, set in `src/data/dataset_utils.py` and implemented in the `self._random_split()` of the ExperimentDataset class), as it will deterministically produce train/val/test splits.                                                 |
| `--model_seed`         | True                          | Seed used to set the Random, numpy and Torch seeds (see `initialize_gpu_seed()` method in `src/helpers/seed_helper.py`)                                                                                                |
| `--use_validation_set` | False                         | Whether to use a train/val/test split or just train/val.                                                                                               |
| `--learning_rate, --adam_eps, --warmup_steps, --weight_decay, --max_grad_norm, --use_softmax_layer`        | -                             | Training parameters, only run with default values during experiments. |
| `--num_epochs`         | 5                             | Number of finetuning iterations to run. In each epoch, all data from the training set is passed through the transformer. Empirically, most models so far trained for 3-5 epochs with the first epoch being best (due to overfitting, etc.) |
| `--save_model`         | -                             | Whether the model checkpoint should be saved locally. Only leave this out for debugging purposes.                                                                                                                 |
| `--save_config`        | -                             | Whether the config of the model should be saved (these parameters among others). Only leave this out for debugging purposes.                                                                                      |
| `--wandb`              | True                          | Whether the progress should be logged to [Weights and Biases (WandB)](https://wandb.com), an experiment analytics platform. Adding this parameter *will* log the experiment.                                          |
| `--no_wandb`           | -                             | Adding this parameter will make sure the experiment is not logged to WandB.  

The training of DistilBERT run in the experiments would be started with:
```python
python src/models/run_transformer_training.py --dataset_name synthetic_companies_small --experiment_name syn_companies_seed_0 --seed 0 --save_model --save_config --no_wandb 
```

This command will train a new DistilBert model on the `synthetic_companies_small` dataset for 5 epochs using a `batch_size` of 8. The training can be logged to [Weights and Biases (WandB)](https://wandb.com), and the config, all of the model checkpoints and predictions on the test split are saved to a folder located at `models/syn_companies_seed_0`. The training is ideally done on a machine with a GPU attached.

### Evaluate candidate pairs with a Transformer model
For transformer models, the blocking and evaluation of candidate pairs is already integrated into TransClean so no additional script needs to be run. See lines 281-307 of `src/matching/matcher.py`.

### Finetune CLER

To train a new CLER model (i.e. 2 instances of RoBERTa, a blocker and a matcher, trained via CLER's methodology), use the script `src/CLER/train.py` with the following arguments:

| Argument               | Default Value         | Description                                                                                   |
|------------------------|-----------------------|-----------------------------------------------------------------------------------------------|
| `--path`              | 'data/'              | The base directory where datasets are stored.                                                |
| `--dataset`           | 'wdc/shoes'          | The name of the dataset to be used for the task.                                             |
| `--run_id`            | 0                    | Identifier for the current run, used to set the seed(s) of the run                           |
| `--batch_size`        | 64                   | Batch size used during training or evaluation.                                               |
| `--max_len`           | 256                  | Maximum toke sequence length the transformer model will be able to process, needs to match the model's input layer size. We use RoBERTa's 256 as a default.    
| `--finetuning`        | False                | **Unused argument (leftover from CLER's implementation)**                                            |                                            |
| `--add_token`         | True                 | Whether to add special COL and VAL tokens to the input sequences.                                        |
| `--lr`                | 3e-5                 | Learning rate for the optimizer.                                                             |
| `--n_epochs`          | 20                   | **Unused argument (leftover from CLER's implementation)**                                                                   |
| `--save_model`        | False                | Whether to save the finetuned model after finetuning.                                            |
| `--logdir`            | 'checkpoints/'       | Directory to store checkpoints and logs.                                                    |
| `--CLlogdir`          | 'CL-sep-sup_0104'    | Directory for storing Contrastive Learning Sentence Embedding weights                                  |
| `--lm`                | 'roberta'            | Language model to use (e.g., `roberta`, `bert`).                                             |
| `--gpu`               | 0                    | Number of available GPUs to use for training or evaluation.                                                    |
| `--fp16`              | False              | Enables mixed precision (fp16) training for faster computation and lower memory usage.       |
| `--total_budget`      | 500                | Total budget for Active Learning during finetuning.                                                 |
| `--warmup_budget`     | 400                | **Unused argument (leftover from CLER's implementation)**                                    |
| `--active_budget`     | 100                | Budget allocated for the Active Learning Stage of CLER training.                                              |
| `--warmup_epochs`     | 20                 | Number of epochs for the warmup phase.                                                       |
| `--topK`              | 5                  | Number of top samples to select during CLER training               |
| `--balance`           | False              | **Unused argument (leftover from CLER's implementation)**                      |
| `--valid_size`        | 200                | **Unused argument (leftover from CLER's implementation)**                                                                  |
| `--blocker_type`      | 'sentbert'           | Type of blocker to use, only used when filtering the test set,                                    |
| `--validation_with_pseudo` | False         | Whether to include pseudo-labeled data in validation.                                        |
| `--aug_type`          | 'random'             | Type of data augmentation to apply, see all options in `CLER/dataset.py`                                        |

The arguments described by "**Unused argument (leftover from CLER's implementation)**" are those added in CLER's [original implementation](https://github.com/wusw14/CLER/tree/master) that are not used beyond the argparse. We choose to leave these arguments in order to alter CLER's code as little as possible. 

The training of CLER run in the experiments would be started with:
```python
python src/CLER/train.py --lr 1e-5 --total_budget 10000 --gpu 1 --dataset synthetic_companies --run_id 0 --batch_size 64 --save_model --topK 10 --logdir syn_comp_ckpt
```
This command will train 2 new CLER models, a blocker and a matcher, on the `synthetic_companies_small` dataset for 20 epochs. Both model's weights will be saved in a folder named as `{--dataset}_CLER_{--total_budget}_{--run_id}` located in `models/`. The training is ideally done on a machine with a GPU attached.


### Evaluate candidate pairs with CLER

The script `src/CLER/test_dynamic.py` runs the blocking and evaluation of candidate pairs with a given CLER run (identified by the `dataset`, `total_budget` and `run_id`). The predictions are saved to a file named `pairwise_matches_preds.csv` but it is necessary to process them because of the following inconsistency:

 - CLER predictions include pairs in both (lid, rid) and (rid, lid) form. If the pairs get different predictions (i.e. Match and NoMatch) this can lead to issues calculating metrics. This issue is ignored in CLER's original implementation as they only account for pairs in the form (lid,rid) since they only ever experiment in a 2-tables setting. 

In order to resolve this inconsistency, we sort all pairs as (lid,rid) with lid < rid and deduplicate them, keeping only the 1st appearance. We do this in `scripts/process_CLER_preds.py` which in the case of a 10k labeling budget it can be run as follows:

```python
python scripts/process_CLER_preds.py --dataset_name synthetic_companies --CLER_experiment_name synthetic_companies_CLER_10000_seed_0 --ground_truth_path data/processed/synthetic_companies/seed_0/test__pre_split__given_matches.csv
```

The file `data/processed/synthetic_companies/seed_0/test__pre_split__given_matches.csv`, containing the ground truth pairs of the test set, is created during the __init__ of the dataset being matched, for example during the training of the transformer model. 

### Running TransClean

To run TransClean (on either a transformer or a CLER run), use the script `scripts/run_TransClean.py` with the following arguments: 

| Argument                            | Default Value          | Description                                                                                           |
|-------------------------------------|------------------------|-------------------------------------------------------------------------------------------------------|
| `--experiment_name`                | None (Required)           | Name of the experiment, same as --experiment_name in `src/models/run_transformer_training.py`                 |
| `--epoch`                          | None (Required)               | Specific epoch of the model produced in --experiment_name to use as initial model for TransClean       |
| `--use_validation_set`             | False                | We add this argument to make sure the test set is not partitioned, see line 74.                                                    |
| `--matcher`                        | None (Required)             | Matcher class to use. Implements blocking and pre_cleanup functionalities and determines the dataset of the experiment.                       |
| `--num_ds`                         | 5                    | Number of data sources to match. Used in the blocking.                                                           |
| `--manual_check`                   | False                | Whether to manually label all edges selected for fine-tuning or weakly label all as 0s.                          |
| `--remove_true_positives`          | False                | Whether to remove true positives from the finetuning edges.                                          |
| `--eval_positive_pairwise_preds`   | False                | Whether to evaluate all positive pairwise predictions with the further finetuned model or just deleted edges. |
| `--threshold`                      | 0.999                | Prob threshold used to determine Match/NoMatch.                                                            |
| `--save_model`                     | False                | Whether to save the model weights of the finetuned model.                                                          |
| `--save_config`                    | False                | Whether to save the configuration used for the experiment.                                           |
| `--finetuning_epochs`              | 3                    | Number of epochs to use for finetuning the model on each iteration of Algorithm 1.                                                    |
| `--finetuning_iterations`          | 5                   | Number of iterations of Algorithm 1.                                                         |

The TransClean run described in the experiments with CLER trained with a 10k labeling budget would be started with:
```python
python scripts/run_TransClean.py --experiment_name synthetic_companies_CLER_10000 --matcher synthetic_companies --threshold 0.999 --manual_check --finetuning_epochs 3 --finetuning_iterations 5 
```

The script will save all results to a series of folders named as `data/results/--dataset_name/--experiment_name/labeling_budget_{budget_value}` for each different labeling budget value (see line 80).

### Evaluation & Visualization

To evaluate the produced matchings after running TransClean, use the script `scripts/get_scores_matching.py` with the following arguments: 

| Argument                     | Default Value     | Description                                                                                  |
|------------------------------|-------------------|----------------------------------------------------------------------------------------------|
| `--dataset_name`             | None (Required)       | Name of the dataset matched.                                                 |
| `--experiment_names_list`    | None (Required)       | List of experiment names to evaluate. Providing multiple values will create the equivalent of Figure 7 in the paper.         |
| `--ground_truth_path`        | None (Required)       | Path to the file containing the ground truth positive pairs of the test split.                                             |
| `--threshold`                | 0.999            | Prob threshold used to determine Match/NoMatch.                                                     |
| `--non_positional_ids`       | False            | The scores are calculated via sparse matrices based on the ids of the records. If these ids are non-positional this should be True, to avoid creating too large matrices. Only necessary in datasets such as WDC Products.                         |
| `--post_edge_recovery`       | False            | Whether to evaluate the Pre or Post TransClean scores.                             |
| `--labeling_budgets_list`    | None             | List of labeling budgets used to produce a graph of labeling budget vs F1 score.            |
| `--add_CLER_scores`          | False            | Adds CLER (Contrastive Learning-based Edge Recovery) scores to the evaluation.              |
| `--CLER_post_cleanup_list`   | None             | List of post-finetuning cleanup steps for different CLER models, used in budget vs F1 graph. |

The following command will produce all the scores and Figure 7 as presented in the paper if all the previous experiments (DistilBERT + CLER with 4 different labeling budgets) have been previously run:  

```python
python scripts/get_scores_matching.py --dataset_name synthetic_companies --experiment_names_list synthetic_companies_initial_training  --ground_truth_path data/processed/synthetic_companies/seed_0/test__pre_split__given_matches.csv --threshold 0.999 --post_edge_recovery --labeling_budgets_list 5000 --labeling_budgets_list 10000 --labeling_budgets_list 50000 --labeling_budgets_list 100000 --add_CLER_scores --CLER_post_cleanup_list synthetic_companies_CLER_5000 --CLER_post_cleanup_list synthetic_companies_CLER_10000 --CLER_post_cleanup_list synthetic_companies_CLER_50000  --CLER_post_cleanup_list synthetic_companies_CLER_100000
```

Finally, in order to produce Figure 6 from the paper (where we visualize TransClean combined with CLER trained on a 10k labeling budget) run the `scripts/visualize_TransClean_cleanup.py` with the following arguments:

```python
python scripts/visualize_TransClean_cleanup.py --dataset_name synthetic_companies --experiment_name synthetic_companies_CLER_10000 --model_name CLER_10000 --labeling_budgets_list 5000 --labeling_budgets_list 10000 --labeling_budgets_list 50000 --labeling_budgets_list 100000 --add_transitive_preds
```

### Setting up a new dataset

Add the dataset to the `Config` dictionary in `src/models/config.py` and implement its Preprocessor, BaseTokenizer and ExperimentDataset subclasses in the respective files.
Running `src/models/run_transformer_training.py` on the new dataset will then generate all the necessary processed files in the `data/processed/<your_dataset_name>` directory.

- Note: This part is automatically done during the training of a new transformer model, if the dataset has not been preprocessed before.
- Note: Files are generated to cache the preprocessed state of the dataset. If you want to make sure the dataset is preprocessed again (i.e. if the pipeline changed), manually delete the files unter `data/processed/<your_dataset_name>/seed_<your_seed>/`

For training CLER, add new data loading functions to `src/CLER/train.py` and `src/CLER/test_dynamic.py`, see examples of the implemented datasets.



