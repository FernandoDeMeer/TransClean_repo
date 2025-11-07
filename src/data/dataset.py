from src.data.dataset_utils import SplitMethod
from src.helpers.seed_helper import init_seed
from src.helpers.logging_helper import setup_logging
from src.data.dataset_utils import *
from src.data.default_benchmark_tokenizer import DefaultBenchmarkTokenizer
from src.data.syn_company_tokenizer import SynCompanyTokenizer
from src.data.preprocessor import  SynCompanyPreprocessor, WDCPreprocessor, CameraPreprocessor, MonitorPreprocessor, MusicBrainzPreprocessor
from src.data.tokenizer import BaseTokenizer
from src.models.config import Config, DEFAULT_NONMATCH_RATIO, DEFAULT_SEED, DEFAULT_SEQ_LENGTH, DEFAULT_TRAIN_FRAC
from src.helpers.path_helper import *
import copy
import os
import sys
import json 
import re

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod


sys.path.append(os.getcwd())


setup_logging()


# Wrapper class for datasets. The goal is to
# have all relevant CSVs accessed through this class, so that
# we do not have to wrangle with files and paths directly,
# but rather get what we need easily.
#
# To add a new dataset, simply add the Config in models/config.py
class ExperimentDataset(ABC):
    # Static method to expose available datasets
    @staticmethod
    def available_datasets():
        return Config.DATASETS.keys()

    @staticmethod
    def create_instance(name: str,
                        model_name: str,
                        use_val: bool = False,
                        split_method: SplitMethod = SplitMethod.RANDOM,
                        seed: int = DEFAULT_SEED,
                        do_lower_case=True,
                        max_seq_length: int = DEFAULT_SEQ_LENGTH,
                        train_frac: float = DEFAULT_TRAIN_FRAC,
                        nonmatch_ratio: int = DEFAULT_NONMATCH_RATIO):

        if name == 'synthetic_companies' or name == 'synthetic_companies_small':
            return SynCompanyDataset(name=name, model_name=model_name, split_method=SplitMethod.PRE_SPLIT, seed=seed,
                                            do_lower_case=do_lower_case, max_seq_length=max_seq_length,
                                            train_frac=train_frac, nonmatch_ratio=nonmatch_ratio, use_val=use_val)
        elif name == 'wdc':
            return WDCBenchmarkDataset(name=name, model_name=model_name, split_method=SplitMethod.PRE_SPLIT, seed=seed,
                                         do_lower_case=do_lower_case, max_seq_length=max_seq_length,
                                         train_frac=train_frac, nonmatch_ratio=nonmatch_ratio, use_val=use_val)    
        elif name == 'camera':
            return CameraDataset(name=name, model_name=model_name, split_method=SplitMethod.PRE_SPLIT, seed=seed,
                                         do_lower_case=do_lower_case, max_seq_length=max_seq_length,
                                         train_frac=train_frac, nonmatch_ratio=nonmatch_ratio, use_val=use_val)
        elif name == 'monitor':
            return MonitorDataset(name=name, model_name=model_name, split_method=SplitMethod.PRE_SPLIT, seed=seed,
                                         do_lower_case=do_lower_case, max_seq_length=max_seq_length,
                                         train_frac=train_frac, nonmatch_ratio=nonmatch_ratio, use_val=use_val)
        elif name == 'musicbrainz':
            return MusicBrainzDataset(name=name, model_name=model_name, split_method=SplitMethod.PRE_SPLIT, seed=seed,
                                         do_lower_case=do_lower_case, max_seq_length=max_seq_length,
                                         train_frac=train_frac, nonmatch_ratio=nonmatch_ratio, use_val=use_val)

        else:
            raise Exception('This dataset has not been implemented.')

    def __init__(self, name: str, model_name: str, use_val: bool,
                 split_method: SplitMethod = SplitMethod.RANDOM,
                 seed: int = DEFAULT_SEED, do_lower_case=True, max_seq_length: int = DEFAULT_SEQ_LENGTH,
                 train_frac: float = DEFAULT_TRAIN_FRAC, nonmatch_ratio: int = DEFAULT_NONMATCH_RATIO):

        self.name = self._check_dataset_name(name)
        self.raw_file_path = dataset_raw_file_path(Config.DATASETS[self.name])
        self.model_name = self._check_model_name(model_name)
        self.seed = seed
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.split_method = split_method
        self.use_val = use_val
        self.train_frac = train_frac
        # "X-1 to 1" ratio nonmatches to matches (for generation)
        self.nonmatch_ratio = nonmatch_ratio
        self.tokenizer = None

        # Set the seed on all libraries
        init_seed(self.seed)
        self.preprocessor = None

    def _check_dataset_name(self, name):
        configured_dataset_names = Config.DATASETS.keys()

        if name not in configured_dataset_names:
            raise ValueError(f':dataset_name {name} should be one of [{configured_dataset_names}]')
        return name

    def _check_model_name(self, model_name):
        configured_model_names = Config.MODELS.keys()

        if model_name not in configured_model_names:
            raise ValueError(f':model_name {model_name} should be one of [{configured_model_names}]')
        return model_name

    def get_split_method_name(self):
        return str(self.split_method).split('.')[1].lower()

    def get_raw_df(self):
        return self.preprocessor.get_raw_df()

    def get_entity_data(self):
        return self.preprocessor.get_entity_data()

    # Generates the tokenized data for every entity in the dataset.
    def get_tokenized_data(self):
        try:
            return self.tokenized_data
        except AttributeError:
            tokenized_file_path = dataset_processed_file_path(self.name, 'tokenized_data__' + self.model_name + '.json',
                                                              seed=self.seed)

            if file_exists_or_create(tokenized_file_path):
                self.tokenized_data = pd.read_json(tokenized_file_path)

            else:
                self.tokenized_data, _ = self.tokenizer.tokenize_df(self.get_entity_data())
                self.tokenized_data.to_json(tokenized_file_path)

        return self.tokenized_data

    # Generates all known positive matches from the raw data
    def get_matches(self):
        try:
            self.matches_df
        except AttributeError:
            processed_matches_path = dataset_processed_file_path(self.name, 'pos_matches.csv', seed=self.seed)

            if file_exists_or_create(processed_matches_path):
                self.matches_df = pd.read_csv(processed_matches_path)
            else:
                self.matches_df = self.get_matches__implementation()
                self.matches_df.to_csv(processed_matches_path, index=False)

        return self.matches_df

    @abstractmethod
    def get_matches__implementation(self):
        raise NotImplementedError("Needs to be implemented on subclass.")

    # randomly assigning matches to train/test
    #
    def _random_split(self):
        def split_fn(df: pd.DataFrame, train_frac: float):
            train_df = df.sample(frac=train_frac, random_state=self.seed)
            test_df = df.drop(train_df.index)
            val_df = pd.DataFrame()
            if self.use_val:
                # split the validation set as half of the test set, i.e.
                # both test and valid sets will be of the same size
                #
                val_df = test_df.sample(frac=0.5, random_state=self.seed)
                test_df = test_df.drop(val_df.index)
            return train_df, test_df, val_df

        return split_fn


    def pre_split(self):
        raise NotImplementedError("Needs to be implemented on subclass.")

    # Separates the matches into train and test
    #
    def _get_train_test_val_given_matches(self, train_frac: float):
        if self.split_method == SplitMethod.RANDOM:
            split_fn = self._random_split()
        elif self.split_method == SplitMethod.PRE_SPLIT:
            split_fn = self.pre_split()
        else:
            raise NotImplementedError(
                f"Split method '{self.split_method}' not implemented. \
                Make sure to include the seed when implementing a new one.")

        try:
            if self.use_val:
                return self.train_given, self.test_given, self.val_given
            else:
                return self.train_given, self.test_given, pd.DataFrame()
        except AttributeError:
            method_name = self.get_split_method_name()
            train_file_path = dataset_processed_file_path(self.name, f'train__{method_name}__given_matches.csv',
                                                          seed=self.seed)
            test_file_path = dataset_processed_file_path(self.name, f'test__{method_name}__given_matches.csv',
                                                         seed=self.seed)
            validation_file_path = dataset_processed_file_path(self.name, f'val__{method_name}__given_matches.csv',
                                                         seed=self.seed)

            check_val = file_exists_or_create(validation_file_path) if self.use_val else True

            if file_exists_or_create(train_file_path) and file_exists_or_create(train_file_path) and check_val:
                self.train_given = pd.read_csv(train_file_path)
                self.test_given = pd.read_csv(test_file_path)
                self.validation_given = pd.read_csv(validation_file_path) if self.use_val else None
            else:
                matches = self.get_matches()
                self.train_given, self.test_given, self.validation_given = split_fn(matches, train_frac)
                self.train_given.to_csv(train_file_path, index=False)
                self.test_given.to_csv(test_file_path, index=False)
                if not self.validation_given.empty:
                    self.validation_given.to_csv(validation_file_path, index=False)

        return self.train_given, self.test_given, self.validation_given

    def get_train_test_val(self):
        try:
            return self.train_df, self.test_df, self.validation_df
        except AttributeError:
            method_name = self.get_split_method_name()
            train_file_path = dataset_processed_file_path(self.name, f'train__{method_name}__all_matches.csv',
                                                          seed=self.seed)
            test_file_path = dataset_processed_file_path(self.name, f'test__{method_name}__all_matches.csv',
                                                         seed=self.seed)
            validation_file_path = dataset_processed_file_path(self.name, f'val__{method_name}__all_matches.csv',
                                                               seed=self.seed)

            check_val = file_exists_or_create(validation_file_path) if self.use_val else True

            if file_exists_or_create(train_file_path) and file_exists_or_create(train_file_path) and check_val:
                self.train_df = pd.read_csv(train_file_path)
                self.test_df = pd.read_csv(test_file_path)
                self.validation_df = pd.read_csv(validation_file_path) if self.use_val else pd.DataFrame()
            else:
                # prebuild given matches self.train_given / self.test_given
                train_given, test_given, validation_given = \
                    self._get_train_test_val_given_matches(train_frac=self.train_frac)

                self.train_df, self.test_df, self.validation_df = \
                    self.get_train_test_val__implementation(train_given, test_given, validation_given)

                self.train_df.to_csv(train_file_path, index=False)
                self.test_df.to_csv(test_file_path, index=False)
                if not self.validation_df.empty:
                    self.validation_df.to_csv(validation_file_path, index=False)

        return self.train_df, self.test_df, self.validation_df

    def get_validation(self):
        """
        reads the validation file if it exists
        does NOT create one if it does not exist yet, unlike train/test set
        """
        try:
            return self.validation_df
        except AttributeError:
            self.validation_df = None
            validation_file_path = dataset_processed_file_path(self.name, 'validation.csv', seed=self.seed)

            if file_exists_or_create(validation_file_path):
                self.validation_df = pd.read_csv(validation_file_path)


            return self.validation_df

    @abstractmethod
    def get_train_test_val__implementation(self, train_given, test_given, val_given):
        # return train_df, test_df, val_df
        raise NotImplementedError("Should be implemented in the respective subclasses.")

    # returns the PyTorch dataloaders ready for use
    def get_data_loaders(self, batch_size: int = 8):
        train_df, test_df, validation_df = self.get_train_test_val()

        train_ds = PytorchDataset(model_name=self.name, idx_df=train_df, data_df=self.get_tokenized_data(),
                                  tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        test_ds = PytorchDataset(model_name=self.name, idx_df=test_df, data_df=self.get_tokenized_data(),
                                 tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
        test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

        if validation_df.empty:
            val_dl = None
        else:
            val_ds = PytorchDataset(model_name=self.name, idx_df=validation_df, data_df=self.get_tokenized_data(),
                                 tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
            val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size)

        return train_dl, test_dl, val_dl


class SynBaseDataset(ExperimentDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.full_ds = False

        if self.name == 'synthetic_companies' or self.name == 'synthetic_companies_small':
            self.ds_type = 'companies'
            if self.name == 'synthetic_companies':
                self.full_ds = True
        else:
            raise ValueError(f'{self.name} does not have a supported synthetic dataset type.')

    def pre_split(self):
        def get_pre_split_train_test_val(matches, train_frac):
            pos_val_matches_path = dataset_raw_file_path(os.path.join('synthetic_data', 'seed_0', self.ds_type,
                                                                      f'{"" if self.full_ds  else "filtered_"}val.csv'))
            pos_val_matches_df = pd.read_csv(pos_val_matches_path, index_col= 0)

            pos_train_matches_path = dataset_raw_file_path(os.path.join('synthetic_data', 'seed_0', self.ds_type,
                                                                        f'{"" if self.full_ds  else "filtered_"}train.csv'))
            pos_train_matches_df = pd.read_csv(pos_train_matches_path, index_col=0)

            pos_test_matches_path = dataset_raw_file_path(os.path.join('synthetic_data', 'seed_0', self.ds_type,
                                                                       'test.csv'))
            pos_test_matches_df = pd.read_csv(pos_test_matches_path, index_col=0)

            return pos_train_matches_df, pos_test_matches_df, pos_val_matches_df
        return get_pre_split_train_test_val

    def get_matches__implementation(self):
        """
        for this class we do not need the file pos_matches, hence we return an empty df
        """
        return pd.DataFrame()

    def get_train_test_val__implementation(self, train_given, test_given, val_given):
        test_df, train_df, val_df = add_random_non_matches_train_val_test(train_given=train_given,
                                      test_given=test_given,
                                      val_given=val_given,
                                      nonmatch_ratio=self.nonmatch_ratio,
                                      name=self.name)
        return train_df, test_df, val_df


class SynCompanyDataset(SynBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = SynCompanyTokenizer(self.model_name, self.do_lower_case, self.max_seq_length)
        self.preprocessor = SynCompanyPreprocessor(self.raw_file_path, self.name, seed=self.seed)


class WDCBenchmarkDataset(ExperimentDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = DefaultBenchmarkTokenizer(self.model_name, self.do_lower_case, self.max_seq_length)
        self.preprocessor = WDCPreprocessor(self.raw_file_path, self.name, seed=self.seed)

    def get_matches__implementation(self):
        """
        For this class we do not need the file pos_matches, hence we return an empty df
        """
        return pd.DataFrame()
        
    
    def get_train_test_val__implementation(self, train_given, test_given, val_given):
        """
        In this class we do not need to add additional negative training samples (they're already in the benchmark)
        """
        return train_given, test_given, val_given
        
    
    def pre_split(self):
        def get_pre_split_train_test_val(matches, train_frac):
            pos_train_matches_path = dataset_raw_file_path(os.path.join('wdc_80pair', 'train.csv'))
            pos_train_matches_df = pd.read_csv(pos_train_matches_path)
            pos_train_matches_df['nexus_id'] = -1  # Add dummy nexus_id column

            pos_val_matches_path = dataset_raw_file_path(os.path.join('wdc_80pair', 'val.csv'))
            pos_val_matches_df = pd.read_csv(pos_val_matches_path)
            pos_val_matches_df['nexus_id'] = -1  # Add dummy nexus_id column

            pos_test_matches_path = dataset_raw_file_path(os.path.join('wdc_80pair', 'test.csv'))
            pos_test_matches_df = pd.read_csv(pos_test_matches_path)
            pos_test_matches_df['nexus_id'] = -1  # Add dummy nexus_id column

            return pos_train_matches_df, pos_test_matches_df, pos_val_matches_df
        
        return get_pre_split_train_test_val


class CameraDataset(ExperimentDataset):
    """
        The camera dataset consists of multiple json files and a csv file containing the
        ground truth matches.

        -camera/camera_ground_truths/camera_entity_resolution_gt.csv 
            - Contains 2 columns: "entity_id,spec_id"
                - "entity_id" is the unique identifier for each entity/record group in the dataset
                - "spec_id" is the unique identifier for each record in the dataset, allows to query
                the record via query_keys = spec_id.split(//) and then querying the json representing 
                the record via camera/camera_specs/query_keys[0]/query_keys[1].json
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = DefaultBenchmarkTokenizer(self.model_name, self.do_lower_case, self.max_seq_length)
        self.preprocessor = CameraPreprocessor(self.raw_file_path, self.name, seed=self.seed)
        self.construct_dataset_from_gt()

    def get_matches__implementation(self):
        """
        For this class we do not need the file pos_matches, hence we return an empty df
        """
        return pd.DataFrame()

    def get_train_test_val__implementation(self, train_given, test_given, val_given):
        test_df, train_df, val_df = add_random_non_matches_train_val_test(train_given=train_given,
                                      test_given=test_given,
                                      val_given=val_given,
                                      nonmatch_ratio=self.nonmatch_ratio,
                                      name=self.name)
        return train_df, test_df, val_df
    

    def pre_split(self):
        def get_pre_split_train_test_val(matches, train_frac):
            pos_train_matches_path = dataset_raw_file_path(os.path.join('camera', 'train.csv'))
            pos_train_matches_df = pd.read_csv(pos_train_matches_path)

            pos_val_matches_path = dataset_raw_file_path(os.path.join('camera', 'val.csv'))
            pos_val_matches_df = pd.read_csv(pos_val_matches_path)

            pos_test_matches_path = dataset_raw_file_path(os.path.join('camera', 'test.csv'))
            pos_test_matches_df = pd.read_csv(pos_test_matches_path)

            return pos_train_matches_df, pos_test_matches_df, pos_val_matches_df

        return get_pre_split_train_test_val

    def construct_dataset_from_gt(self):
        """
        Constructs the dataset from the ground truth matches
        """
        # Check if the dataset already exists
        dataset_file_path = dataset_raw_file_path(os.path.join('camera', 'camera.csv'))
        matches_file_path = dataset_raw_file_path(os.path.join('camera', 'train.csv'))
        if file_exists_or_create(dataset_file_path) and file_exists_or_create(matches_file_path):
            return
        
        # Load the ground truth matches
        gt_matches_path = dataset_raw_file_path(os.path.join('camera', 'camera_ground_truths', 'camera_entity_resolution_gt.csv'))
        gt_matches_df = pd.read_csv(gt_matches_path)
        
        # Create new DataFrames to store the dataset
        records_df = pd.DataFrame(columns=['page_title', 'text'])
        train_matches_df = pd.DataFrame(columns=['lid', 'rid', 'label'])
        val_matches_df = pd.DataFrame(columns=['lid', 'rid', 'label'])
        test_matches_df = pd.DataFrame(columns=['lid', 'rid', 'label'])

        # Iterate through the ground truth csv and construct the dataset

        # We want to split the entities into train/val/test in a 60/20/20 split

        entity_ids = gt_matches_df['entity_id'].unique()
        # Split the entity_ids into train/val/test
        num_entities = len(entity_ids)
        num_train = int(num_entities * .6)
        num_val = int(num_entities * .2)
        num_test = num_entities - num_train - num_val
        train_entity_ids = entity_ids[:num_train]
        val_entity_ids = entity_ids[num_train:num_train + num_val]
        test_entity_ids = entity_ids[num_train + num_val:]
        # Create a mapping from entity_id to split
        entity_id_to_split = {}
        for entity_id in train_entity_ids:
            entity_id_to_split[entity_id] = 'train'
        for entity_id in val_entity_ids:
            entity_id_to_split[entity_id] = 'val'
        for entity_id in test_entity_ids:
            entity_id_to_split[entity_id] = 'test'

        for entity_id, group in tqdm(gt_matches_df.groupby('entity_id'), desc='Processing entity_id groups'):
            # Get the spec_ids for the current entity_id
            spec_ids = group['spec_id'].tolist()

            # Get the records for the current entity_id

            new_ids = []

            for i in range(len(spec_ids)):
                # Load the json file for the current spec_id
                spec_id = spec_ids[i]
                json_path = dataset_raw_file_path(os.path.join('camera', 'camera_specs', spec_id.split('//')[0], spec_id.split('//')[1] + '.json'))
                with open(json_path, 'r') as f:
                    record = json.load(f)
                    # Store the page_title and put all the other attributes into the text column
                    records_df = records_df.append({'page_title': record['<page title>'], 'text': ' '.join([f'{k}: {v}' for k, v in record.items() if k != '<page title>'])}, ignore_index=True)
                    new_ids.append(len(records_df) - 1)

            # Create matches of the form (lid, rid) for the current entity_id
            for i in range(len(spec_ids)):
                for j in range(i + 1, len(spec_ids)):
                    if entity_id_to_split[entity_id] == 'train':
                        train_matches_df = train_matches_df.append({'lid': new_ids[i], 'rid': new_ids[j], 'label': 1}, ignore_index=True)
                    elif entity_id_to_split[entity_id] == 'val':
                        val_matches_df = val_matches_df.append({'lid': new_ids[i], 'rid': new_ids[j], 'label': 1}, ignore_index=True)
                    elif entity_id_to_split[entity_id] == 'test':
                        test_matches_df = test_matches_df.append({'lid': new_ids[i], 'rid': new_ids[j], 'label': 1}, ignore_index=True)
        

        # Save the constructed dataset
        dataset_file_path = dataset_raw_file_path(os.path.join('camera', 'camera.csv'))
        records_df.to_csv(dataset_file_path, index=True, header= True)
        matches_file_path = dataset_raw_file_path(os.path.join('camera', 'train.csv'))
        train_matches_df.to_csv(matches_file_path, index=False)
        matches_file_path = dataset_raw_file_path(os.path.join('camera', 'val.csv'))
        val_matches_df.to_csv(matches_file_path, index=False)
        matches_file_path = dataset_raw_file_path(os.path.join('camera', 'test.csv'))
        test_matches_df.to_csv(matches_file_path, index=False)

class MonitorDataset(ExperimentDataset):
    """
        The monitor dataset consists of multiple json files and a csv file containing the
        ground truth matches.

        -monitor/monitor_ground_truths/monitor_entity_resolution_gt.csv 
            - Contains 2 columns: "entity_id,spec_id"
                - "entity_id" is the unique identifier for each entity/record group in the dataset
                - "spec_id" is the unique identifier for each record in the dataset, allows to query
                the record via query_keys = spec_id.split(//) and then querying the json representing 
                the record via monitor/monitor_specs/query_keys[0]/query_keys[1].json
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = DefaultBenchmarkTokenizer(self.model_name, self.do_lower_case, self.max_seq_length)
        self.preprocessor = MonitorPreprocessor(self.raw_file_path, self.name, seed=self.seed)
        self.construct_dataset_from_gt()

    def get_matches__implementation(self):
        """
        For this class we do not need the file pos_matches, hence we return an empty df
        """
        return pd.DataFrame()

    def get_train_test_val__implementation(self, train_given, test_given, val_given):
        test_df, train_df, val_df = add_random_non_matches_train_val_test(train_given=train_given,
                                      test_given=test_given,
                                      val_given=val_given,
                                      nonmatch_ratio=self.nonmatch_ratio,
                                      name=self.name)
        return train_df, test_df, val_df
    

    def pre_split(self):
        def get_pre_split_train_test_val(matches, train_frac):
            pos_train_matches_path = dataset_raw_file_path(os.path.join('monitor', 'train.csv'))
            pos_train_matches_df = pd.read_csv(pos_train_matches_path)

            pos_val_matches_path = dataset_raw_file_path(os.path.join('monitor', 'val.csv'))
            pos_val_matches_df = pd.read_csv(pos_val_matches_path)

            pos_test_matches_path = dataset_raw_file_path(os.path.join('monitor', 'test.csv'))
            pos_test_matches_df = pd.read_csv(pos_test_matches_path)

            return pos_train_matches_df, pos_test_matches_df, pos_val_matches_df

        return get_pre_split_train_test_val

    def construct_dataset_from_gt(self):
        """
        Constructs the dataset from the ground truth matches
        """
        # Check if the dataset already exists
        dataset_file_path = dataset_raw_file_path(os.path.join('monitor', 'monitor.csv'))
        matches_file_path = dataset_raw_file_path(os.path.join('monitor', 'train.csv'))
        if file_exists_or_create(dataset_file_path) and file_exists_or_create(matches_file_path):
            return
        
        # Load the ground truth matches
        gt_matches_path = dataset_raw_file_path(os.path.join('monitor', 'monitor_ground_truths', 'monitor_entity_resolution_gt.csv'))
        gt_matches_df = pd.read_csv(gt_matches_path)
        
        # Create new DataFrames to store the dataset
        records_df = pd.DataFrame(columns=['page_title', 'text'])
        train_matches_df = pd.DataFrame(columns=['lid', 'rid', 'label'])
        val_matches_df = pd.DataFrame(columns=['lid', 'rid', 'label'])
        test_matches_df = pd.DataFrame(columns=['lid', 'rid', 'label'])

        # Iterate through the ground truth csv and construct the dataset

        # We want to split the entities into train/val/test in a 60/20/20 split

        entity_ids = gt_matches_df['entity_id'].unique()
        # Split the entity_ids into train/val/test
        num_entities = len(entity_ids)
        num_train = int(num_entities * .6)
        num_val = int(num_entities * .2)
        num_test = num_entities - num_train - num_val
        train_entity_ids = entity_ids[:num_train]
        val_entity_ids = entity_ids[num_train:num_train + num_val]
        test_entity_ids = entity_ids[num_train + num_val:]
        # Create a mapping from entity_id to split
        entity_id_to_split = {}
        for entity_id in train_entity_ids:
            entity_id_to_split[entity_id] = 'train'
        for entity_id in val_entity_ids:
            entity_id_to_split[entity_id] = 'val'
        for entity_id in test_entity_ids:
            entity_id_to_split[entity_id] = 'test'

        for entity_id, group in tqdm(gt_matches_df.groupby('entity_id'), desc='Processing entity_id groups'):
            # Get the spec_ids for the current entity_id
            spec_ids = group['spec_id'].tolist()

            # Get the records for the current entity_id

            new_ids = []

            for i in range(len(spec_ids)):
                # Load the json file for the current spec_id
                spec_id = spec_ids[i]
                json_path = dataset_raw_file_path(os.path.join('monitor', 'monitor_specs', spec_id.split('//')[0], spec_id.split('//')[1] + '.json'))
                with open(json_path, 'r') as f:
                    record = json.load(f)
                    # Store the page_title and put all the other attributes into the text column
                    records_df = records_df.append({'page_title': record['<page title>'], 'text': ' '.join([f'{k}: {v}' for k, v in record.items() if k != '<page title>'])}, ignore_index=True)
                    new_ids.append(len(records_df) - 1)

            # Create matches of the form (lid, rid) for the current entity_id
            for i in range(len(spec_ids)):
                for j in range(i + 1, len(spec_ids)):
                    if entity_id_to_split[entity_id] == 'train':
                        train_matches_df = train_matches_df.append({'lid': new_ids[i], 'rid': new_ids[j], 'label': 1}, ignore_index=True)
                    elif entity_id_to_split[entity_id] == 'val':
                        val_matches_df = val_matches_df.append({'lid': new_ids[i], 'rid': new_ids[j], 'label': 1}, ignore_index=True)
                    elif entity_id_to_split[entity_id] == 'test':
                        test_matches_df = test_matches_df.append({'lid': new_ids[i], 'rid': new_ids[j], 'label': 1}, ignore_index=True)
        

        # Save the constructed dataset
        dataset_file_path = dataset_raw_file_path(os.path.join('monitor', 'monitor.csv'))
        records_df.to_csv(dataset_file_path, index=True, header= True)
        matches_file_path = dataset_raw_file_path(os.path.join('monitor', 'train.csv'))
        train_matches_df.to_csv(matches_file_path, index=False)
        matches_file_path = dataset_raw_file_path(os.path.join('monitor', 'val.csv'))
        val_matches_df.to_csv(matches_file_path, index=False)
        matches_file_path = dataset_raw_file_path(os.path.join('monitor', 'test.csv'))
        test_matches_df.to_csv(matches_file_path, index=False)

class MusicBrainzDataset(ExperimentDataset):
    """
    
    The MusicBrainz dataset contains on a single csv of records belonging to 5 different sources.

        Each record contains: 

        TID: a unique record's id (in the complete dataset).
        CID: cluster id (records having the same CID are duplicate)
        CTID: a unique id within a cluster (if two records belong to the same cluster they will have the same CID but different CTIDs). These ids (CTID) start with 1 and grow until cluster size.
        SourceID: identifies to which source a record belongs (there are five sources). The sources are deduplicated.
        Id: the original id from the source. Each source has its own Id-Format. Uniqueness is not guaranteed!! (can be ignored).
        number: track or song number in the album.
        length: the length of the track.
        artist: the interpreter (artist or band) of the track.
        year: date of publication.
        language: language of the track.
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = DefaultBenchmarkTokenizer(self.model_name, self.do_lower_case, self.max_seq_length)
        self.preprocessor = MusicBrainzPreprocessor(self.raw_file_path, self.name, seed=self.seed)
        self.construct_dataset_from_gt()

    def get_matches__implementation(self):
        """
        For this class we do not need the file pos_matches, hence we return an empty df
        """
        return pd.DataFrame()

    def get_train_test_val__implementation(self, train_given, test_given, val_given):
        test_df, train_df, val_df = add_random_non_matches_train_val_test(train_given=train_given,
                                      test_given=test_given,
                                      val_given=val_given,
                                      nonmatch_ratio=self.nonmatch_ratio,
                                      name=self.name)
        return train_df, test_df, val_df
    
    def pre_split(self):
        def get_pre_split_train_test_val(matches, train_frac):
            pos_train_matches_path = dataset_raw_file_path(os.path.join('musicbrainz', 'train.csv'))
            pos_train_matches_df = pd.read_csv(pos_train_matches_path)

            pos_val_matches_path = dataset_raw_file_path(os.path.join('musicbrainz', 'val.csv'))
            pos_val_matches_df = pd.read_csv(pos_val_matches_path)

            pos_test_matches_path = dataset_raw_file_path(os.path.join('musicbrainz', 'test.csv'))
            pos_test_matches_df = pd.read_csv(pos_test_matches_path)

            return pos_train_matches_df, pos_test_matches_df, pos_val_matches_df

        return get_pre_split_train_test_val

    def construct_dataset_from_gt(self):
        """
        Constructs the dataset from the ground truth matches
        """
        # Check if the dataset already exists
        dataset_file_path = dataset_raw_file_path(os.path.join('musicbrainz', 'musicbrainz.csv'))
        matches_file_path = dataset_raw_file_path(os.path.join('musicbrainz', 'train.csv'))
        if file_exists_or_create(dataset_file_path) and file_exists_or_create(matches_file_path):
            return
        
        # Load the ground truth matches
        original_df = pd.read_csv(dataset_raw_file_path(os.path.join('musicbrainz', 'musicbrainz-20-A01.csv')))

        pairs = []
        # Split the 'CID' values into train/val/test with a 60/20/20 split

        # Get the unique cluster ids
        cluster_ids = original_df['CID'].unique()
        # Split the cluster ids into train/val/test
        num_entities = len(cluster_ids)
        num_train = int(num_entities * .6)
        num_val = int(num_entities * .2)
        train_cluster_ids = cluster_ids[:num_train]
        val_cluster_ids = cluster_ids[num_train:num_train + num_val]
        test_cluster_ids = cluster_ids[num_train + num_val:]
        # Create a mapping from cluster_id to split
        cluster_id_to_split = {}
        for cluster_id in train_cluster_ids:
            cluster_id_to_split[cluster_id] = 'train'
        for cluster_id in val_cluster_ids:
            cluster_id_to_split[cluster_id] = 'val'
        for cluster_id in test_cluster_ids:
            cluster_id_to_split[cluster_id] = 'test'
        # Iterate through the original dataframe and create the pairs

        train_pairs = []
        val_pairs = []
        test_pairs = []
        # Iterate through the original dataframe and create the pairs
        for cluster_id, group in tqdm(original_df.groupby('CID'), desc='Processing CID groups'):
            # Get the ids for each record of the current cluster
            spec_ids = group.index.tolist()
            
            for i in range(len(spec_ids)):
                for j in range(i + 1, len(spec_ids)):
                    if cluster_id_to_split[cluster_id] == 'train':
                        train_pairs.append((spec_ids[i], spec_ids[j], 1))
                    elif cluster_id_to_split[cluster_id] == 'val':
                        val_pairs.append((spec_ids[i], spec_ids[j], 1))
                    elif cluster_id_to_split[cluster_id] == 'test':
                        test_pairs.append((spec_ids[i], spec_ids[j], 1))
        
        # Create the DataFrames for the pairs
        train_matches_df = pd.DataFrame(train_pairs, columns=['lid', 'rid', 'label'])
        val_matches_df = pd.DataFrame(val_pairs, columns=['lid', 'rid', 'label'])
        test_matches_df = pd.DataFrame(test_pairs, columns=['lid', 'rid', 'label'])

        # Create the DataFrame for the records, first drop all the id columns

        records_df = original_df.drop(columns=['TID', 'CID', 'CTID', 'SourceID', 'id'])
        records_df.to_csv(dataset_file_path, header= True)
        # Save the pairs
        matches_file_path = dataset_raw_file_path(os.path.join('musicbrainz', 'train.csv'))
        train_matches_df.to_csv(matches_file_path, index=False)
        matches_file_path = dataset_raw_file_path(os.path.join('musicbrainz', 'val.csv'))
        val_matches_df.to_csv(matches_file_path, index=False)
        matches_file_path = dataset_raw_file_path(os.path.join('musicbrainz', 'test.csv'))
        test_matches_df.to_csv(matches_file_path, index=False)



class PytorchDataset(Dataset):
    def __init__(self, model_name: str, idx_df: pd.DataFrame, data_df: pd.DataFrame, tokenizer: BaseTokenizer,
                 max_seq_length: int):
        self.model_name = model_name
        self.idx_df = idx_df
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # self.label_list = sorted(self.idx_df.label.unique())

    def __len__(self):
        return len(self.idx_df)

    def __getitem__(self, idx):
        row = self.idx_df.iloc[idx]

        l_txt = copy.deepcopy(self.data_df.loc[row['lid'], 'tokenized'])
        r_txt = copy.deepcopy(self.data_df.loc[row['rid'], 'tokenized'])
        label = row['label']

        seq = self.tokenizer.generate_sample(l_txt, r_txt, label)

        # Also return the initial IDs for easier logging
        raw_batch = (
            torch.tensor([row['lid']], dtype=torch.long),
            torch.tensor([row['rid']], dtype=torch.long),
        )

        return seq + raw_batch
