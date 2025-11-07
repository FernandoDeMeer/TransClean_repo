import os
import pickle
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from src.data import full_data_utils
from src.data.dataset import ExperimentDataset
from src.helpers.path_helper import *
from src.models.config import Config
from src.models.pytorch_model import PyTorchModel
from src.matching.matcher import Matcher



class CompanyMatcher(Matcher, ABC):
    def __init__(self, id_attributes, number_of_candidates: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.number_of_candidates = number_of_candidates
        self.id_attributes = id_attributes

    def get_matching_securities(self, company, raw_sec_df) -> pd.DataFrame:
        associated_securities = raw_sec_df[(raw_sec_df['issuer_id'] == company['external_id']) & (
            raw_sec_df['data_source_id'] == company['data_source_id'])]
        if len(associated_securities) == 0:
            return associated_securities
        identifiers = dict()
        for identifier in self.id_attributes.values():
            id_values = associated_securities[identifier].dropna().unique()
            try:
                identifiers[identifier] = [] if len(id_values) == 0 else id_values
            except TypeError:
                identifiers[identifier] = []
        # Check for securities that share any of the identifiers in other data sources
        matching_securities = raw_sec_df[
            (
                (raw_sec_df[self.id_attributes['isin']].isin(identifiers[self.id_attributes['isin']]))
                | (raw_sec_df[self.id_attributes['cusip']].isin(identifiers[self.id_attributes['cusip']]))
                | (raw_sec_df[self.id_attributes['valor']].isin(identifiers[self.id_attributes['valor']]))
                | (raw_sec_df[self.id_attributes['sedol']].isin(identifiers[self.id_attributes['sedol']]))
            ) &
            (raw_sec_df['data_source_id'] != company['data_source_id'])
        ]
        return matching_securities

class SynCompanyMatcher(CompanyMatcher):
    def __init__(self, **kwargs):
        super().__init__(id_attributes=full_data_utils.ID_ATTRIBUTES, **kwargs)

    def get_test_entity_data(self) -> pd.DataFrame:
        # Check if the test records have already been previously saved
        test_folder_path = os.path.join(self.processed_folder_path, 'test_entity_data.csv')
        if file_exists_or_create(test_folder_path):
            return pd.read_csv(test_folder_path)

        syn_data_path = os.path.join('data', 'raw', 'synthetic_data', 'seed_0')
        companies_mapping_df_path = os.path.join(syn_data_path, 'companies_master_mapping_seed_0.csv')
        mapping_df = pd.read_csv(companies_mapping_df_path)
        split = full_data_utils.get_train_val_test_split(mapping_df)

        syn_records_file_path = os.path.join(syn_data_path, 'synthetic_records_dicts_seed_0.pkl')
        with open(syn_records_file_path, 'rb') as f:
            syn_records_dicts = pickle.load(f)
        f.close()

        syn_records_test = [dict for dict in syn_records_dicts if dict['gen_id'] in split['test']]
        test_records = []
        for dict in syn_records_test:
            test_records.append(dict['company_records'])

        companies_data_test_df = pd.concat(test_records)
        companies_data_test_df = companies_data_test_df.sort_values(['data_source_id', 'external_id'],
                                                                    ascending=[True, True])
        companies_data_test_df = companies_data_test_df.drop('gen_id', axis=1)

        # We need to set the id column for the test records according to the raw data
        companies_data_test_df = full_data_utils.add_id_to_records(companies_data_test_df,
                                                                   raw_df=self.model.dataset.get_raw_df())

        # Save the test records
        companies_data_test_df.to_csv(test_folder_path, index=False)

        return companies_data_test_df

    def blocking(self, test_entity_data: pd.DataFrame) -> pd.DataFrame:

        candidates_path = os.path.join(self.processed_folder_path, 'full_test_candidates.csv')
        if file_exists_or_create(candidates_path):
            return pd.read_csv(candidates_path)

        master_mapping_df = full_data_utils.load_syn_master_mapping(dataset_name=self.model.dataset.name)
        master_mapping_df = master_mapping_df.astype(str)

        # join the gen_id into the test_entity_data dataframe for easier lookup lateron
        test_entity_data = test_entity_data.merge(master_mapping_df.astype(
            int), left_on=['external_id', 'data_source_id'], right_on=['external_id', 'data_source_id'])

        candidate_pairs = []
        # We load the securities dataset to get id_overlap pairs
        securities_raw_file_path = dataset_raw_file_path(
            os.path.join('synthetic_data', 'seed_0', 'synthetic_securities_dataset_seed_0_size_984942_sorted.csv'))
        securities_data_df = pd.read_csv(securities_raw_file_path)

        for index, company in tqdm(test_entity_data.iterrows(), total=test_entity_data.shape[0],
                                   desc='Gathering id_overlap candidate pairs'):

            matching_securities = self.get_matching_securities(company, securities_data_df)

            if len(matching_securities) == 0:
                continue
            else:
                # Find the matching companies based on the matching securities
                company_candidate_ids = test_entity_data.merge(matching_securities[['issuer_id', 'data_source_id']],
                                                               left_on=['external_id', 'data_source_id'],
                                                               right_on=['issuer_id', 'data_source_id'])['id'].unique()

                # Assign the label and finalize candidate pair
                left_gen_id = company['gen_id']
                for idx in company_candidate_ids:
                    if idx == company['id']:
                        continue

                    right_gen_id = test_entity_data[test_entity_data['id'] == idx]['gen_id'].item()

                    label = 1 if left_gen_id == right_gen_id else 0
                    candidate_pairs.append((company['id'], idx, label, 'id_overlap'))

        indicators, tokenized_test_records = self.get_tknzd_records_and_overlap_indicators(test_entity_data)

        # Looping over the records to find the most similar (most overlapping tokens) records
        for i, (record_id, tokenized_record) in tqdm(enumerate(tokenized_test_records.iterrows()), total=tokenized_test_records.shape[0], desc='Finding closest records'):
            top_overlap_idx = self.get_top_overlap_idx(i, indicators, test_entity_data)

            # Assign the label and finalize candidate pair
            left_gen_id = test_entity_data.iloc[i]['gen_id']
            for idx in top_overlap_idx:
                if idx == i:
                    continue

                right_record = test_entity_data.iloc[idx]
                right_gen_id = right_record['gen_id']

                label = 1 if left_gen_id == right_gen_id else 0
                candidate_pairs.append((record_id, right_record['id'], label, 'text_match'))

        test_candidates_df = pd.DataFrame(data=candidate_pairs, columns=['lid', 'rid', 'label', 'match_type'])

        self.save_test_candidates(test_candidates_df)

        return test_candidates_df
    
class WDCMatcher(Matcher):
    def __init__(self, number_of_candidates: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.number_of_candidates = number_of_candidates

    def get_test_entity_data(self) -> pd.DataFrame:
        # Check if the test records have already been previously saved
        test_folder_path = os.path.join(self.processed_folder_path, 'test_entity_data.csv')
        if file_exists_or_create(test_folder_path):
            return pd.read_csv(test_folder_path)

        wdc_data_path = os.path.join('data', 'raw', 'wdc_80pair')
        wdc_test_pairs_file_path = os.path.join(wdc_data_path, 'test.csv')
        self.wdc_test_pairs = pd.read_csv(wdc_test_pairs_file_path)
        wdc_records_file_path = os.path.join(wdc_data_path, 'wdc_80pair.csv')
        wdc_records = pd.read_csv(wdc_records_file_path)

        # Filter the test records 
        test_records = wdc_records[(wdc_records['id'].isin(self.wdc_test_pairs['lid'])) | (wdc_records['id'].isin(self.wdc_test_pairs['rid']))]
        
        # Save the test records
        test_records.to_csv(test_folder_path, index=False)
        return test_records

    def blocking(self, test_entity_data: pd.DataFrame) -> pd.DataFrame:
        candidates_path = os.path.join(self.processed_folder_path, 'full_test_candidates.csv')
        if file_exists_or_create(candidates_path):
            return pd.read_csv(candidates_path)
        
        # Check if self.wdc_test_pairs is already loaded
        if not hasattr(self, 'wdc_test_pairs'):
            wdc_data_path = os.path.join('data', 'raw', 'wdc_80pair')
            wdc_test_pairs_file_path = os.path.join(wdc_data_path, 'test.csv')
            self.wdc_test_pairs = pd.read_csv(wdc_test_pairs_file_path)

        candidate_pairs = []

        indicators, tokenized_test_records = self.get_tknzd_records_and_overlap_indicators(test_entity_data)

        # Looping over the records to find the most similar (most overlapping tokens) records

        for i, (record_id, tokenized_record) in tqdm(enumerate(tokenized_test_records.iterrows()), total=tokenized_test_records.shape[0],
                                                      desc='Finding closest records via token overlap'):
            top_overlap_idx = self.get_top_overlap_idx_one_source(i, indicators, test_entity_data)

            top_overlap_ids = test_entity_data.iloc[top_overlap_idx]['id'].values

            # Assign the label and finalize candidate pair. First check the pairs where record_id appears (either as lid or rid)
            record_pairs = self.wdc_test_pairs[(self.wdc_test_pairs['lid'] == record_id) | (self.wdc_test_pairs['rid'] == record_id)]
            for idx in top_overlap_ids:
                if idx == record_id:
                    continue

                # Check if the pair is in the record_pairs
                if len(record_pairs[(record_pairs['lid'] == idx) | (record_pairs['rid'] == idx)]) > 0:
                    pair = record_pairs[(record_pairs['lid'] == idx) | (record_pairs['rid'] == idx)]
                    if pair['label'].item() == 1:
                        label = 1
                    else:
                        label = 0
                else:
                    label = 0


                candidate_pairs.append((record_id, idx, label, 'text_match'))

        test_candidates_df = pd.DataFrame(data=candidate_pairs, columns=['lid', 'rid', 'label', 'match_type'])

        self.save_test_candidates(test_candidates_df)

        return test_candidates_df
    

class CameraMatcher(Matcher):
    def __init__(self, number_of_candidates: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.number_of_candidates = number_of_candidates

    def get_test_entity_data(self) -> pd.DataFrame:
        # Check if the test records have already been previously saved
        test_folder_path = os.path.join(self.processed_folder_path, 'test_entity_data.csv')
        if file_exists_or_create(test_folder_path):
            return pd.read_csv(test_folder_path)

        camera_data_path = os.path.join('data', 'raw', 'camera')
        camera_test_pairs_file_path = os.path.join(camera_data_path, 'test.csv')
        self.camera_test_pairs = pd.read_csv(camera_test_pairs_file_path)
        camera_records_file_path = os.path.join(camera_data_path, 'camera.csv')
        camera_records = pd.read_csv(camera_records_file_path)
        # Rename the first column to 'id' 
        camera_records.rename(columns={camera_records.columns[0]: 'id'}, inplace=True)

        # Filter the test records 
        test_records = camera_records[(camera_records['id'].isin(self.camera_test_pairs['lid'])) | (camera_records['id'].isin(self.camera_test_pairs['rid']))]
        
        # Save the test records
        test_records.to_csv(test_folder_path, index=False)
        return test_records
    
    def blocking(self, test_entity_data: pd.DataFrame) -> pd.DataFrame:
        candidates_path = os.path.join(self.processed_folder_path, 'full_test_candidates.csv')
        if file_exists_or_create(candidates_path):
            return pd.read_csv(candidates_path)

        # Check if self.camera_test_pairs is already loaded
        if not hasattr(self, 'camera_test_pairs'):
            camera_data_path = os.path.join('data', 'raw', 'camera')
            camera_test_pairs_file_path = os.path.join(camera_data_path, 'test.csv')
            self.camera_test_pairs = pd.read_csv(camera_test_pairs_file_path)
        
        candidate_pairs = []

        indicators, tokenized_test_records = self.get_tknzd_records_and_overlap_indicators(test_entity_data)

        # Looping over the records to find the most similar (most overlapping tokens) records

        for i, (record_id, tokenized_record) in tqdm(enumerate(tokenized_test_records.iterrows()), total=tokenized_test_records.shape[0],
                                                      desc='Finding closest records via token overlap'):
            top_overlap_idx = self.get_top_overlap_idx_one_source(i, indicators, test_entity_data)

            top_overlap_ids = test_entity_data.iloc[top_overlap_idx]['id'].values

            # Assign the label and finalize candidate pair. First check the pairs where record_id appears (either as lid or rid)
            record_pairs = self.camera_test_pairs[(self.camera_test_pairs['lid'] == record_id) | (self.camera_test_pairs['rid'] == record_id)]
            for idx in top_overlap_ids:
                if idx == record_id:
                    continue # Skip the same record

                # Check if the pair is in the record_pairs
                if len(record_pairs[(record_pairs['lid'] == idx) | (record_pairs['rid'] == idx)]) > 0:
                    pair = record_pairs[(record_pairs['lid'] == idx) | (record_pairs['rid'] == idx)]
                    if pair['label'].item() == 1:
                        label = 1
                    else:
                        label = 0
                else:
                    label = 0


                candidate_pairs.append((record_id, idx, label, 'text_match'))
        
        test_candidates_df = pd.DataFrame(data=candidate_pairs, columns=['lid', 'rid', 'label', 'match_type'])
        self.save_test_candidates(test_candidates_df)
        return test_candidates_df
    
class MonitorMatcher(Matcher):
    def __init__(self, number_of_candidates: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.number_of_candidates = number_of_candidates

    def get_test_entity_data(self) -> pd.DataFrame:
        # Check if the test records have already been previously saved
        test_folder_path = os.path.join(self.processed_folder_path, 'test_entity_data.csv')
        if file_exists_or_create(test_folder_path):
            return pd.read_csv(test_folder_path)

        monitor_data_path = os.path.join('data', 'raw', 'monitor')
        monitor_test_pairs_file_path = os.path.join(monitor_data_path, 'test.csv')
        self.monitor_test_pairs = pd.read_csv(monitor_test_pairs_file_path)
        monitor_records_file_path = os.path.join(monitor_data_path, 'monitor.csv')
        monitor_records = pd.read_csv(monitor_records_file_path)
        # Rename the first column to 'id'
        monitor_records.rename(columns={monitor_records.columns[0]: 'id'}, inplace=True)

        # Filter the test records 
        test_records = monitor_records[(monitor_records['id'].isin(self.monitor_test_pairs['lid'])) | (monitor_records['id'].isin(self.monitor_test_pairs['rid']))]
        
        # Save the test records
        test_records.to_csv(test_folder_path, index=False)
        return test_records
    
    def blocking(self, test_entity_data: pd.DataFrame) -> pd.DataFrame:
        candidates_path = os.path.join(self.processed_folder_path, 'full_test_candidates.csv')
        if file_exists_or_create(candidates_path):
            return pd.read_csv(candidates_path)

        # Check if self.monitor_test_pairs is already loaded
        if not hasattr(self, 'monitor_test_pairs'):
            monitor_data_path = os.path.join('data', 'raw', 'monitor')
            monitor_test_pairs_file_path = os.path.join(monitor_data_path, 'test.csv')
            self.monitor_test_pairs = pd.read_csv(monitor_test_pairs_file_path)
        
        candidate_pairs = []

        indicators, tokenized_test_records = self.get_tknzd_records_and_overlap_indicators(test_entity_data)

        # Looping over the records to find the most similar (most overlapping tokens) records

        for i, (record_id, tokenized_record) in tqdm(enumerate(tokenized_test_records.iterrows()), total=tokenized_test_records.shape[0],
                                                      desc='Finding closest records via token overlap'):
            top_overlap_idx = self.get_top_overlap_idx_one_source(i, indicators, test_entity_data)

            top_overlap_ids = test_entity_data.iloc[top_overlap_idx]['id'].values

            # Assign the label and finalize candidate pair. First check the pairs where record_id appears (either as lid or rid)

            record_pairs = self.monitor_test_pairs[(self.monitor_test_pairs['lid'] == record_id) | (self.monitor_test_pairs['rid'] == record_id)]
            for idx in top_overlap_ids:
                if idx == record_id:
                    continue # Skip the same record

                # Check if the pair is in the record_pairs
                if len(record_pairs[(record_pairs['lid'] == idx) | (record_pairs['rid'] == idx)]) > 0:
                    pair = record_pairs[(record_pairs['lid'] == idx) | (record_pairs['rid'] == idx)]
                    if pair['label'].item() == 1:
                        label = 1
                    else:
                        label = 0
                else:
                    label = 0


                candidate_pairs.append((record_id, idx, label, 'text_match'))

        test_candidates_df = pd.DataFrame(data=candidate_pairs, columns=['lid', 'rid', 'label', 'match_type'])
        self.save_test_candidates(test_candidates_df)
        return test_candidates_df
    
class MusicBrainzMatcher(Matcher):
    def __init__(self, number_of_candidates: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.number_of_candidates = number_of_candidates

    def get_test_entity_data(self) -> pd.DataFrame:
        # Check if the test records have already been previously saved
        test_folder_path = os.path.join(self.processed_folder_path, 'test_entity_data.csv')
        if file_exists_or_create(test_folder_path):
            return pd.read_csv(test_folder_path)

        musicbrainz_data_path = os.path.join('data', 'raw', 'musicbrainz')
        musicbrainz_test_pairs_file_path = os.path.join(musicbrainz_data_path, 'test.csv')
        self.musicbrainz_test_pairs = pd.read_csv(musicbrainz_test_pairs_file_path)
        musicbrainz_records_file_path = os.path.join(musicbrainz_data_path, 'musicbrainz.csv')
        musicbrainz_records = pd.read_csv(musicbrainz_records_file_path)
        # Rename the first column to 'id'
        musicbrainz_records.rename(columns={musicbrainz_records.columns[0]: 'id'}, inplace=True)

        # Filter the test records 
        test_records = musicbrainz_records[(musicbrainz_records['id'].isin(self.musicbrainz_test_pairs['lid'])) | (musicbrainz_records['id'].isin(self.musicbrainz_test_pairs['rid']))]
        
        # Save the test records
        test_records.to_csv(test_folder_path, index=False)
        return test_records
    
    def blocking(self, test_entity_data: pd.DataFrame) -> pd.DataFrame:
        candidates_path = os.path.join(self.processed_folder_path, 'full_test_candidates.csv')
        if file_exists_or_create(candidates_path):
            return pd.read_csv(candidates_path)

        # Check if self.musicbrainz_test_pairs is already loaded
        if not hasattr(self, 'musicbrainz_test_pairs'):
            musicbrainz_data_path = os.path.join('data', 'raw', 'musicbrainz')
            musicbrainz_test_pairs_file_path = os.path.join(musicbrainz_data_path, 'test.csv')
            self.musicbrainz_test_pairs = pd.read_csv(musicbrainz_test_pairs_file_path)
        
        candidate_pairs = []

        indicators, tokenized_test_records = self.get_tknzd_records_and_overlap_indicators(test_entity_data)

        # Looping over the records to find the most similar (most overlapping tokens) records

        for i, (record_id, tokenized_record) in tqdm(enumerate(tokenized_test_records.iterrows()), total=tokenized_test_records.shape[0],
                                                      desc='Finding closest records via token overlap'):
            top_overlap_idx = self.get_top_overlap_idx_one_source(i, indicators, test_entity_data)

            top_overlap_ids = test_entity_data.iloc[top_overlap_idx]['id'].values

            # Assign the label and finalize candidate pair. First check the pairs where record_id appears (either as lid or rid)

            record_pairs = self.musicbrainz_test_pairs[(self.musicbrainz_test_pairs['lid'] == record_id) | (self.musicbrainz_test_pairs['rid'] == record_id)]
            for idx in top_overlap_ids:
                if idx == record_id:
                    continue # Skip the same record

                # Check if the pair is in the record_pairs
                if len(record_pairs[(record_pairs['lid'] == idx) | (record_pairs['rid'] == idx)]) > 0:
                    pair = record_pairs[(record_pairs['lid'] == idx) | (record_pairs['rid'] == idx)]
                    if pair['label'].item() == 1:
                        label = 1
                    else:
                        label = 0
                else:
                    label = 0


                candidate_pairs.append((record_id, idx, label, 'text_match'))

        test_candidates_df = pd.DataFrame(data=candidate_pairs, columns=['lid', 'rid', 'label', 'match_type'])
        self.save_test_candidates(test_candidates_df)
        return test_candidates_df