import os
import pickle
from abc import ABC, abstractmethod
import copy
import torch

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import full_data_utils
from src.data.dataset import ExperimentDataset
from src.helpers.path_helper import *
from src.models.config import Config
from src.matching.LLM_labeling import label_record_pair_with_LLM
from src.CLER.dataset import GTDatasetWithLabelWeights, GTDatasetWithLabel
from src.CLER.runner import train_matcher
from src.CLER.utils import *


class Matcher(ABC):

    def __init__(self, model, processed_folder_path: str = None, results_path: str = None, num_ds: int = 5, subgraph_size_threshold: int = 50):
        self.model = model
        if processed_folder_path:
            self.processed_folder_path = processed_folder_path
        else:
            self.processed_folder_path = dataset_processed_folder_path(dataset_name=self.model.dataset.name)

        if results_path:
            self.results_path = results_path
        else:
            self.results_path = dataset_results_folder_path__with_subfolders(subfolder_list=[self.model.dataset.name, self.model.args.experiment_name])
        
        self.num_ds = num_ds
        self.subgraph_size_threshold = subgraph_size_threshold


    def test_records_from_positive_matches(self, test_entity_data: pd.DataFrame):
        """
        Filter the given test_entity_data down to the records that are actually
        part of the positive matches in the test set, i.e. removing all the records
        that are only part of it because they are one half of a negative match.
        """
        test_df = self.model.dataset.test_df
        test_df = test_df[test_df['label'] == 1]

        # all the unique values of 'lid' and 'rid' in test_df
        test_ids = set()
        test_ids.update(list(test_df['lid']))
        test_ids.update(list(test_df['rid']))

        test_records = test_entity_data[test_entity_data['id'].isin(test_ids)]
        return test_records

    @abstractmethod
    def blocking(self, test_entity_data: pd.DataFrame) -> pd.DataFrame:
        """
        takes the full test_entity_data and creates candidate pairs from it
        """

        raise NotImplementedError("Should be implemented in the respective subclasses.")

    def get_test_entity_data(self) -> pd.DataFrame:
        """
        basic implementation to get test_entity_data
        can be overwritten if needed by the respective subclass
        """
        # Check if the test records have already been previously saved
        test_folder_path = os.path.join(self.processed_folder_path, 'test_entity_data.csv')
        if file_exists_or_create(test_folder_path):
            return pd.read_csv(test_folder_path)

        test_id_df = self.model.dataset.test_df

        full_entity_data = self.model.dataset.get_entity_data()

        test_ids = set()
        test_ids.update(list(test_id_df['lid']))
        test_ids.update(list(test_id_df['rid']))

        test_entity_data = full_entity_data[full_entity_data['id'].isin(test_ids)]
        # Save the test_entity_data
        test_entity_data.to_csv(os.path.join(self.processed_folder_path, 'test_entity_data.csv'), index=False)

        return test_entity_data

    def save_test_candidates(self, candidates_df: pd.DataFrame):
        """
        saves a blocked candidates_df with cols (lid, rid, label, match_type) in the processed path of the ds
        """
        # Delete the rows with lid == rid
        candidates_df = candidates_df[candidates_df['lid'] != candidates_df['rid']]
        # Delete duplicates
        candidates_df = candidates_df.drop_duplicates(subset=['lid', 'rid'])
        # Save the test candidates
        candidates_df.to_csv(os.path.join(self.processed_folder_path, 'full_test_candidates.csv'), index=False)

    def load_and_save_pairwise_matches_preds(self, args, candidate_df):
        """
        Loads the pairwise_matches_preds from the prediction_log and saves them to the processed folder
        """
        file_name = "".join([self.model.args.model_name, '__prediction_log__ep', str(args.epoch), '.csv'])
        log_path = experiment_file_path(args.experiment_name, file_name)

        pairwise_matches_preds = pd.read_csv(log_path)
        # Add the match_type column to the pairwise_matches_preds
        pairwise_matches_preds = pairwise_matches_preds.merge(candidate_df[['lid', 'rid', 'match_type']], left_on=['lids', 'rids'], right_on=['lid', 'rid'])
        pairwise_matches_preds = pairwise_matches_preds.drop(columns=['labels', 'predictions', 'lid', 'rid'])
        # Rename the column 'prediction_proba' to 'prob'
        pairwise_matches_preds = pairwise_matches_preds.rename(columns={'prediction_proba': 'prob'})
        # Rename the lids and rids columns to lid and rid
        pairwise_matches_preds = pairwise_matches_preds.rename(columns={'lids': 'lid', 'rids': 'rid'})

        # Save the pairwise_matches_preds
        pairwise_matches_preds.to_csv(os.path.join(self.results_path, 'pairwise_matches_preds.csv'), index=False)

        return pairwise_matches_preds
    
    ###########################################################################

    # TransClean Implementation

    ###########################################################################

    def run_TransClean(self, args):
        """
        Runs the whole TransClean pipeline:
        - A) blocking
        - B) pairwise matching
        Cleanup with finetuning:
        - C) finetuning dataset construction (w/ manual labeling & optional XAI attributions) via the neg_transitive_predictions_heuristic
        - D) finetuning + transitive edges prediction w/ finetuned model
        - E) Update the pairwise_matches_preds with the new finetuned model, go back to C) for another finetuning iteration if we have not reached the max number of iterations
        - F) Post-finetuning graph cleanup
        - G) Post cleanup checks
        Edge recovery:
        - H) For every deleted edge, we evaluate its transitive edges if it was added back to the matches_graph. If all of them are predicted as matches, we add the edge back to the matches_graph,
        otherwise (unless all of them are predicted as non-matches) we manually check the edge.
        """
        #Check if threshold is set in args
        if hasattr(args, 'threshold'):
            self.threshold = args.threshold
        else:
            self.threshold=0.999
        self.args = args
        self.finetune_cleanup_dict = {'negative_transitive_preds': [], 'positive_transitive_preds': [],
                                'true_positives': [], 'false_positives': [],
                                'removed_true_positives_transitive': [], 'removed_false_positives_transitive': [],
                                'labeled_pairs': []}

        # Load the ground truth from the processed folder 
        if 'CLER' in self.model.args.model_name:
            self.ground_truth = pd.read_csv(os.path.join(dataset_raw_file_path(Config.SPLITS[self.model.dataset.name]), 'test.csv'))
        else:
            _ = self.blocking(self.get_test_entity_data())
            self.ground_truth = pd.read_csv(os.path.join(self.processed_folder_path, 'full_test_candidates.csv'))
        self.ground_truth = self.sort_lids_rids(self.ground_truth)

        self.finetuning_iteration = 1
        self.finetune_results_path = update_results_path(self.results_path, ['labeling_budget_{}'.format(self.args.labeling_budget), 'finetune_iteration_{}'.format(self.finetuning_iteration)])
        _ = self.first_finetuning_graph_cleanup_iteration()

        while self.finetuning_iteration < self.args.finetuning_iterations:
            # Update the results path
            self.finetuning_iteration += 1
            self.finetune_results_path = update_results_path(self.results_path, ['labeling_budget_{}'.format(self.args.labeling_budget), 'finetune_iteration_{}'.format(self.finetuning_iteration)])
            if self.check_results_folder(self.finetuning_iteration):
                # This iteration has already been run, we load the model, the pairwise_matches_preds and the finetuning_edges
                model_path = os.path.join(self.finetune_results_path, '{}_finetune_iteration_{}.pt'.format(self.model.args.model_name, self.finetuning_iteration))
                self.model.load(model_path)
                if os.path.exists(os.path.join(self.finetune_results_path, 'finetune_dict.csv')):
                    self.model.finetune_dict = pd.read_csv(os.path.join(self.finetune_results_path, 'finetune_dict.csv')).to_dict(orient='list')
                self.pairwise_matches_preds = pd.read_csv(os.path.join(self.finetune_results_path, 'pairwise_matches_preds.csv'))
                self.finetuning_edges = pd.read_csv(os.path.join(self.finetune_results_path, 'finetuning_edges.csv'))
                self.finetune_cleanup_dict = pd.read_csv(os.path.join(self.finetune_results_path, 'finetune_cleanup_dict.csv')).to_dict(orient='list')
            else:
                if not os.path.exists(self.finetune_results_path):
                    os.makedirs(self.finetune_results_path)
                self.finetuning_graph_cleanup_iteration()
        
        print('Finished finetuning after {} iterations'.format(self.finetuning_iteration))
        self.post_finetuning_results_path = update_results_path(self.results_path, ['labeling_budget_{}'.format(self.args.labeling_budget)])

        # F) Post-finetuning graph cleanup

        self.pairwise_matches_preds = self.post_finetuning_cleanup()

        # G) We now perform post-finetuning checks
        post_finetuning_effort = self.post_finetuning_checks()

        # Generate again the matches_graph with the transitive matches
        matches_graph = full_data_utils.generate_matches_graph(self.pairwise_matches_preds, threshold=self.threshold)

        matches_graph, _ = full_data_utils.generate_transitive_matches_graph(matches_graph,
                                                                                add_transitive_edges=True,
                                                                                results_path= self.finetune_results_path,
                                                                                subgraph_size_threshold=self.subgraph_size_threshold)        
        # Save the self.model's finetune_dict as a csv
        if hasattr(self.model, 'finetune_dict'):
            finetune_dict_df = pd.DataFrame(self.model.finetune_dict)
            finetune_dict_df.to_csv(os.path.join(self.finetune_results_path, 'finetune_dict.csv'), index=False)

        # Save the manual check count
        self.manual_check_count_true_pos = self.pairwise_matches_preds['manual_check_label'].value_counts().get(1)
        if self.manual_check_count_true_pos is None:
            self.manual_check_count_true_pos = 0
        self.manual_check_count_false_pos = self.pairwise_matches_preds['manual_check_label'].value_counts().get(0)
        if self.manual_check_count_false_pos is None:
            self.manual_check_count_false_pos = 0
        self.unchecked_pairs_count = self.pairwise_matches_preds['manual_check_label'].value_counts().get(-1)
        if self.unchecked_pairs_count is None:
            self.unchecked_pairs_count = 0
        manual_check_count_df = pd.DataFrame({'manual_check_count_true_pos': [self.manual_check_count_true_pos]})
        manual_check_count_df['manual_check_count_false_pos'] = [self.manual_check_count_false_pos]
        manual_check_count_df['unchecked_pairs_count'] = [self.unchecked_pairs_count]
        manual_check_count_df['percent_checked'] = [(self.manual_check_count_true_pos + self.manual_check_count_false_pos) / (self.manual_check_count_true_pos + self.manual_check_count_false_pos + self.unchecked_pairs_count) * 100]
        manual_check_count_df['post_finetuning_effort'] = [post_finetuning_effort]
        manual_check_count_df['number_of_initial_pairwise_matches'] = [len(self.pairwise_matches_preds)]
        manual_check_count_df.to_csv(os.path.join(self.finetune_results_path, 'manual_check_count.csv'), index=False)

        # Save the finetune_cleanup_dict
        finetune_cleanup_dict_df = pd.DataFrame(self.finetune_cleanup_dict)
        finetune_cleanup_dict_df.to_csv(os.path.join(self.finetune_results_path, 'finetune_cleanup_dict.csv'), index=False)

        # H) Start the edge recovery process
        self.edge_recovery(self.remaining_labeling_budget)
        print('Finished Matching with Finetuning and Edge Recovery')

    ############################################################################

    # Finetuning Graph Cleanup

    ############################################################################

    def first_finetuning_graph_cleanup_iteration(self,):

        if self.check_results_folder(self.finetuning_iteration):
            # This iteration has already been run, we load the model, the pairwise_matches_preds and the finetuning_edges
            model_path = os.path.join(self.finetune_results_path, '{}_finetune_iteration_{}.pt'.format(self.model.args.model_name, self.finetuning_iteration))

            self.model.load(model_path)

            if os.path.exists(os.path.join(self.finetune_results_path, 'finetune_dict.csv')):
                self.model.finetune_dict = pd.read_csv(os.path.join(self.finetune_results_path, 'finetune_dict.csv')).to_dict(orient='list')
            self.pairwise_matches_preds = pd.read_csv(os.path.join(self.finetune_results_path, 'pairwise_matches_preds.csv'))
            self.finetuning_edges = pd.read_csv(os.path.join(self.finetune_results_path, 'finetuning_edges.csv'))
            self.finetune_cleanup_dict = pd.read_csv(os.path.join(self.finetune_results_path, 'finetune_cleanup_dict.csv')).to_dict(orient='list')

            if not "CLER" in self.model.args.model_name:
                # We modify the train_data_loader of the model to be the finetuning_edges 
                self.model.train_data_loader.dataset.idx_df = self.finetuning_edges[['lid', 'rid', 'label']]
                self.model.test_data_loader.dataset.idx_df = self.pairwise_matches_preds[self.pairwise_matches_preds['prob'] >= self.threshold][['lid', 'rid', 'label']]
            # Exit the function
            return None

        elif "CLER" in self.model.args.model_name:
            if not os.path.exists(self.finetune_results_path):
                os.makedirs(self.finetune_results_path)
            # Load the pairwise_matches_preds
            pairwise_matches_preds_path = os.path.join(self.results_path, 'pairwise_matches_preds.csv')
            self.pairwise_matches_preds = pd.read_csv(pairwise_matches_preds_path) 

        else:
            if not os.path.exists(self.finetune_results_path):
                os.makedirs(self.finetune_results_path)

            test_entity_data = self.get_test_entity_data()

            # Get candidates using the blocking function
            candidate_df = self.blocking(test_entity_data)

            # drop match_type for now to use the testing function
            candidate_idx_df = candidate_df.drop(columns=['match_type'])

            # inject the candidates into the test_data_loader, not the best way, but quick for now
            self.model.test_data_loader.dataset.idx_df = candidate_idx_df

            # First check if the pairwise_matches_preds have already been previously saved
            pairwise_matches_preds_path = os.path.join(self.results_path, 'pairwise_matches_preds.csv')

            if file_exists_or_create(pairwise_matches_preds_path):
                self.pairwise_matches_preds = pd.read_csv(pairwise_matches_preds_path)
            else:
                # Run the pairwise matching
                self.model.test(epoch=self.args.epoch) 
                self.pairwise_matches_preds = self.load_and_save_pairwise_matches_preds(self.args, candidate_df)

        # Generate the matches_graph and get the pre_cleanup transitive matches

        matches_graph = full_data_utils.generate_matches_graph(self.pairwise_matches_preds, threshold=self.threshold)

        _, transitive_matches = full_data_utils.generate_transitive_matches_graph(matches_graph,
                                                                                add_transitive_edges=False,
                                                                                results_path=self.results_path,
                                                                                subgraph_size_threshold=self.subgraph_size_threshold,
                                                                                )

        # Save the pre-cleanup transitive matches
        transitive_matches_df = pd.DataFrame(transitive_matches, columns=['lid', 'rid', 'match_type'])
        if os.path.exists(self.finetune_results_path):
            transitive_matches_df.to_csv(os.path.join(self.results_path, 'pre_cleanup_transitive_matches.csv'), index=False)

        # Add label to the pairwise_matches_preds
        self.pairwise_matches_preds = self.sort_lids_rids(self.pairwise_matches_preds)
        self.pairwise_matches_preds = self.filter_pairwise_matches_preds(self.pairwise_matches_preds, self.threshold)
        self.pairwise_matches_preds = self.pairwise_matches_preds.merge(self.ground_truth[['lid', 'rid', 'label']], on=['lid', 'rid'], how='left').drop_duplicates(subset=['lid', 'rid'])
        if 'CLER' in self.model.args.model_name:
            self.pairwise_matches_preds['label'] = self.pairwise_matches_preds['label'].fillna(0)
            
        self.pairwise_matches_preds.reset_index(drop=True, inplace=True)

        assert self.pairwise_matches_preds['label'].isnull().sum() == 0, 'Some pairwise matches do not have a label'

        # Break down the too large subgraphs
        self.pairwise_matches_preds = self.remove_large_subgraphs(self.pairwise_matches_preds, subgraph_size_threshold=self.subgraph_size_threshold)

        # C) finetuning dataset construction via the neg_transitive_predictions_heuristic   
        self.finetuning_edges = self.create_finetuning_dataset()
        if len(self.finetuning_edges) == 0:
            print('There are too few edges to create a finetuning dataset')
            return None
        else:
            self.finetuning_edges.to_csv(os.path.join(self.finetune_results_path, 'finetuning_edges.csv'), index=False)

        # D) finetuning + pairwise matching w/ finetuned model

        # Make sure there are no NaN values in the finetuning_edges
        self.finetuning_edges = self.finetuning_edges.dropna(subset=['lid', 'rid', 'label'])

        if 'CLER' in self.model.args.model_name:
            self.finetune_CLER_model(self.finetuning_edges)

        else:

            # We modify the train_data_loader of the model to be the finetuning_edges 
            self.model.train_data_loader.dataset.idx_df = self.finetuning_edges[['lid', 'rid', 'label']]
            self.model.test_data_loader.dataset.idx_df = self.pairwise_matches_preds[self.pairwise_matches_preds['prob'] >= self.threshold][['lid', 'rid', 'label']]

            self.finetune_model(self.finetuning_edges, self.finetune_results_path, self.finetuning_iteration)

        # E) Update the pairwise_matches_preds with the new finetuned model

        self.pairwise_matches_preds = self.update_pairwise_matches_preds(self.pairwise_matches_preds)
        self.pairwise_matches_preds.to_csv(os.path.join(self.finetune_results_path, 'pairwise_matches_preds.csv'), index=False)
        finetune_cleanup_df = pd.DataFrame(self.finetune_cleanup_dict)
        finetune_cleanup_df.to_csv(os.path.join(self.finetune_results_path, 'finetune_cleanup_dict.csv'), index=False)

        return None

    def finetuning_graph_cleanup_iteration(self,):
        # C) Create the finetuning dataset via the neg_transitive_predictions_heuristic
        new_finetuning_edges = self.create_finetuning_dataset()
        if len(new_finetuning_edges) > 0:
            self.finetuning_edges = pd.concat([self.finetuning_edges, new_finetuning_edges], ignore_index=True)
            self.finetuning_edges = self.finetuning_edges.drop_duplicates(subset=['lid', 'rid'])
            self.finetuning_edges.to_csv(os.path.join(self.finetune_results_path, 'finetuning_edges.csv'), index=False)
        else:
            print('There are no new candidate edges to add to the finetuning dataset')
            return True, self.pairwise_matches_preds

        # D) finetuning + pairwise matching w/ finetuned model

        # Make sure there are no NaN values in the finetuning_edges
        self.finetuning_edges = self.finetuning_edges.dropna(subset=['lid', 'rid', 'label'])


        if 'CLER' in self.model.args.model_name:
            self.finetune_CLER_model(self.finetuning_edges)
        
        else:
            self.finetune_model(self.finetuning_edges, self.finetune_results_path, self.finetuning_iteration)

        # E) check for stopping criterion, if not met, go back to C)
        self.pairwise_matches_preds = self.update_pairwise_matches_preds(self.pairwise_matches_preds)
        self.pairwise_matches_preds.to_csv(os.path.join(self.finetune_results_path, 'pairwise_matches_preds.csv'), index=False)
        finetune_cleanup_dict_df = pd.DataFrame(self.finetune_cleanup_dict)
        finetune_cleanup_dict_df.to_csv(os.path.join(self.finetune_results_path, 'finetune_cleanup_dict.csv'), index=False)

    ###########################################################################

    # Finetuning Edge Selection Heuristics

    ###########################################################################   

    def create_finetuning_dataset(self):
        """
        Creates the finetuning dataset via the neg_transitive_predictions_heuristic, taking into account the labeling budget if necessary.
        """

        if self.args.labeling_budget is not None:
            self.labeling_budget = self.args.labeling_budget
            self.budget_per_iteration = int(self.labeling_budget / (self.args.finetuning_iterations * 2))  

        # Get the current matches_graph
        self.matches_graph = full_data_utils.generate_matches_graph(self.pairwise_matches_preds, threshold=self.threshold)

        if self.args.labeling_budget is not None:
            finetuning_edges = self.neg_transitive_predictions_heuristic(self.matches_graph, budget=self.budget_per_iteration)
        else:
            finetuning_edges = self.neg_transitive_predictions_heuristic(self.matches_graph)
                
        if len(finetuning_edges) > 0:
            finetuning_edges['finetuning_iteration'] = self.finetuning_iteration
            finetuning_edges = self.sort_lids_rids(finetuning_edges)
        else:
            return finetuning_edges


        if self.args.manual_check:  
            # Perform a "manual labeling" of finetuning_edges and set the labels of all "checked" edges in the pairwise_matches_preds (both true positives and false positives)
            finetuning_edges, self.pairwise_matches_preds = self.manual_check(finetuning_edges, self.pairwise_matches_preds, self.args.remove_true_positives)

        else:
            # In the case of no manual labeling, we use an LLM to label the finetuning edges
            finetuning_edges, self.pairwise_matches_preds = self.LLM_labeling_finetuning(finetuning_edges, self.pairwise_matches_preds)


        return finetuning_edges
                
    def neg_transitive_predictions_heuristic(self, matches_graph, budget=None):
        """
        Selects edges to manually check based on the predictions of the transitive edges of each subgraph. 
        We select edges from the subgraphs with the highest amount of negative transitive predictions.
        From each subgraph we select:

        1. The minimum edge cut of the subgraph (most critical edges in terms of connectivity)
        2. Shortest paths between the nodes of a subset (at most 10) of the negatively predicted transitive edges (if an edge is predicted as a negative, then it is likely that there will be a false positive among the edges of the shortest path)

        """

        if hasattr(self, 'transitive_matches_preds'):
            transitive_matches_preds = self.transitive_matches_preds
        else:
            transitive_matches_preds = self.evaluate_transitive_edges(matches_graph) # Evaluate the transitive matches of the matches_graph (only in the first finetuning epoch)

        subgraphs = list(nx.connected_components(matches_graph)) # The matches_graph doesn't contain the transitive edges
        subgraphs = [matches_graph.subgraph(c) for c in subgraphs]

        negative_transitive_preds = transitive_matches_preds[transitive_matches_preds['prob'] < self.threshold] 
        # We add the negative_transitive_preds to each subgraph
        subgraphs_with_transitive_preds = []
        for subgraph in tqdm(subgraphs, desc='[Neg Transitive Preds Heuristic]', total=len(subgraphs), leave=False):
            subgraph_nodes = set(subgraph.nodes())
            subgraph_negative_transitive_preds = negative_transitive_preds[negative_transitive_preds['lid'].isin(subgraph_nodes) & negative_transitive_preds['rid'].isin(subgraph_nodes)]
            if len(subgraph_negative_transitive_preds) > 0:
                subgraphs_with_transitive_preds.append((subgraph, subgraph_negative_transitive_preds))

        # We sort the subgraphs by the amount of negative transitive predictions
        subgraphs_with_transitive_preds = sorted(subgraphs_with_transitive_preds, key=lambda x: x[1].shape[0], reverse=True)

        edges_to_check = pd.DataFrame()
        for subgraph, neg_transitive_edges  in tqdm(subgraphs_with_transitive_preds, desc='[Neg Transitive Preds Heuristic]', total=len(subgraphs)):

            subgraph_edges = pd.DataFrame(subgraph.edges(data=True), columns=['lid', 'rid', 'match_type'])
            subgraph_edges = pd.concat([subgraph_edges.drop(['match_type'], axis=1), subgraph_edges['match_type'].apply(pd.Series)], axis=1)
            
            if len(neg_transitive_edges) > 0:
                subgraph_edges = subgraph_edges[subgraph_edges['match_type'] != 'transitive_match']
                subgraph_edges = subgraph_edges.drop(columns=['prob', 'match_type'])
                subgraph_edges = self.sort_lids_rids(subgraph_edges)

                # We add the minimum-edge-cut edges of the subgraph to the edges_to_check
                subgraph = nx.from_edgelist(subgraph_edges[['lid', 'rid']].values)
                min_edge_cut = nx.minimum_edge_cut(subgraph)
                min_edge_cut = pd.DataFrame(min_edge_cut, columns=['lid', 'rid'])

                # We add shortest paths between the nodes of a subset of negatively predicted transitive edges
                
                # Select a random subset of 10 edges from the neg_transitive_edges
                neg_transitive_edges = neg_transitive_edges.sample(n=min(len(neg_transitive_edges), 10), random_state=42)

                random_paths_edges = pd.DataFrame(columns=['lid', 'rid'])

                for pair in neg_transitive_edges.iterrows():
                    lid = pair[1]['lid']
                    rid = pair[1]['rid']
                    shortest_path = nx.shortest_path(subgraph, source=lid, target=rid)
                    shortest_path_edges = pd.DataFrame(columns=['lid', 'rid'])
                    for i in range(len(shortest_path) - 1):
                        shortest_path_edges = shortest_path_edges.append({'lid': shortest_path[i], 'rid': shortest_path[i + 1]}, ignore_index=True)
                    
                    random_paths_edges = pd.concat([random_paths_edges, shortest_path_edges], ignore_index=True)

                selected_edges = pd.concat([min_edge_cut, random_paths_edges], ignore_index=True)
                selected_edges = self.sort_lids_rids(selected_edges)
                selected_edges = selected_edges.merge(self.pairwise_matches_preds, on=['lid', 'rid'], how='left')
                assert selected_edges['prob'].isnull().sum() == 0, 'Some subgraph edges do not have a prob'

                edges_to_check = pd.concat([edges_to_check, selected_edges], ignore_index=True)
                edges_to_check = edges_to_check.drop_duplicates(subset=['lid', 'rid'])

            if budget is not None and len(edges_to_check) > 0:
                if len(edges_to_check) >= budget:
                    break
            
        if len(edges_to_check) == 0:
            return pd.DataFrame()
        else:
            if budget is not None:
                if budget < len(edges_to_check):
                    edges_to_check = edges_to_check.iloc[:budget]

        return edges_to_check
    
    ###########################################################################

    # Finetuning Cleanup Utils

    ##########################################################################

    def sort_lids_rids(self, input_df):
        """
        In any of the dfs we have, the lids and rids can appear  either (lid, rid) or (rid, lid) because of the graph representation (order of the vertices in the edge tuple is not guaranteed)

        In order to ease merges, we sort the ids in ascending order
        """
        df = copy.deepcopy(input_df)
        # Check if the df is empty
        if df.empty:
            return df
        else:
            df['lid'], df['rid'] = zip(*df.apply(lambda x: (min(x['lid'], x['rid']), max(x['lid'], x['rid'])), axis=1))
            return df

    def remove_large_subgraphs(self, pairwise_matches_preds, subgraph_size_threshold=50):
        """
        Removes from the matching all subgraphs bigger than the subgraph_size_threshold
        """
        pairwise_matches_preds['pairwise_id'] = pairwise_matches_preds.index
        subgraphs = self.update_subgraphs(pairwise_matches_preds)

        # We break down the exceedingly large subgraphs simply by setting the probs of half of their edges (chosen randomly) to 0
        while len(subgraphs[0]) > subgraph_size_threshold:
            print('[LARGE SUBGRAPHS BREAKDOWN] Largest subgraph size: {}'.format(len(subgraphs[0])))
            subgraph = subgraphs[0]
            random_edges = pd.DataFrame(subgraph.edges(data=True), columns=['lid', 'rid', 'match_type'])
            # Set the probs of the edges in the subgraph to 0
            subgraph_edges = pd.DataFrame(subgraph.edges(data=True), columns=['lid', 'rid', 'match_type'])
            subgraph_edges = self.sort_lids_rids(subgraph_edges)
            subgraph_edges = subgraph_edges.merge(pairwise_matches_preds, on=['lid', 'rid'], how='left')
            subgraph_edges_indices = subgraph_edges['pairwise_id']
            pairwise_matches_preds.loc[subgraph_edges_indices, 'prob'] = 0
            # Update the subgraphs
            subgraphs = self.update_subgraphs(pairwise_matches_preds)
                
        pairwise_matches_preds = pairwise_matches_preds.drop(columns=['pairwise_id'])
        return pairwise_matches_preds



    def update_subgraphs(self, pairwise_matches_preds):
        matches_graph = full_data_utils.generate_matches_graph(pairwise_matches_preds, threshold=self.threshold)
        subgraphs = list(nx.connected_components(matches_graph))
        subgraphs = [matches_graph.subgraph(c) for c in subgraphs]
        subgraphs = sorted(subgraphs, key=lambda x: len(x), reverse=True)
        return subgraphs
    
    def load_LLM_model(self):
        # Initialize the LLM model if needed
        model_name_or_path = "TheBloke/deepseek-llm-7B-base-AWQ"
        self.LLM_model_max_length = 4096

        self.LLM_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.LLM_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            device_map="cuda:0"
        )


    def LLM_labeling_finetuning(self, finetuning_edges, pairwise_matches_preds):
        """
        Use an LLM to label the finetuning edges and set the labels of all "checked" edges in the pairwise_matches_preds
        """
        # Load the LLM model
        self.load_LLM_model()

        if not hasattr(self.model.dataset, 'tokenized_data'):
            self.model.dataset.tokenized_data = self.model.dataset.get_tokenized_data()

        pairwise_matches_preds['pairwise_id'] = pairwise_matches_preds.index

        # Check first if the pairwise_matches_preds have a 'manual_check_label' column
        if 'manual_check_label' not in pairwise_matches_preds.columns:
            pairwise_matches_preds['manual_check_label'] = -1 # -1 indicates that the edge has not been manually checked yet

        # Merge the finetuning_edges with the pairwise_matches_preds
        edges_to_label = finetuning_edges.merge(pairwise_matches_preds[['lid','rid','pairwise_id']], on=['lid', 'rid'], how='left')

        # Now call the LLM to label each of the edges_to_label
        for index, row in tqdm(edges_to_label.iterrows(), desc="[LLM Labeling]", total=len(edges_to_label), leave=False):
            # Check that the edge has not been labelled before
            if pairwise_matches_preds.loc[row['pairwise_id'], 'manual_check_label'] != -1:
                continue
            
            lid_record = self.model.dataset.tokenized_data.loc[row['lid']]['concat_data']
            rid_record = self.model.dataset.tokenized_data.loc[row['rid']]['concat_data']
            # Call the LLM to label the edge
            LLM_label = label_record_pair_with_LLM(self.LLM_model, self.LLM_tokenizer, self.LLM_model_max_length, lid_record, rid_record)
            # Set the label of the edge in the pairwise_matches_preds
            pairwise_matches_preds.loc[row['pairwise_id'], 'manual_check_label'] = LLM_label
            # Set the label of the edge in the finetuning_edges
            finetuning_edges.loc[index, 'finetuning_label'] = LLM_label
            finetuning_edges.loc[index, 'manual_check_label'] = LLM_label

        # Delete the LLM model from the GPU to free up memory

        del self.LLM_model

        return finetuning_edges, pairwise_matches_preds


    def manual_check(self, finetuning_edges, pairwise_matches_preds, remove_true_positives=True):
        """
        Simulate a manual check of the finetuning edges and set the labels of the "checked" edges in the pairwise_matches_preds and the finetuning_edges
        """
        finetuning_edges['finetuning_label'] = finetuning_edges['label']
        finetuning_edges['manual_check_label'] = finetuning_edges['label']
        pairwise_matches_preds['pairwise_id'] = pairwise_matches_preds.index


        # Check first if the pairwise_matches_preds have a 'manual_check_label' column
        if 'manual_check_label' not in pairwise_matches_preds.columns:
            pairwise_matches_preds['manual_check_label'] = -1 # -1 indicates that the edge has not been manually checked yet

        # Merge the finetuning_edges with the pairwise_matches_preds
        checked_edges = finetuning_edges.merge(pairwise_matches_preds[['lid','rid','pairwise_id']], on=['lid', 'rid'], how='left')
        
        # Now set the manual_check_label of the checked edges in the pairwise_matches_pres to be the same as their label
        checked_indices = checked_edges['pairwise_id'].dropna().astype(int)
        pairwise_matches_preds.loc[checked_indices, 'manual_check_label'] = pairwise_matches_preds.loc[checked_indices, 'label']

        if remove_true_positives:
            # Remove the true positives from the finetuning_edges
            finetuning_edges = finetuning_edges[finetuning_edges['label'] == 0]
        
        pairwise_matches_preds = pairwise_matches_preds.drop(columns=['pairwise_id'])

        assert pairwise_matches_preds['manual_check_label'].isnull().sum() == 0, 'The manual_check_label column in the pairwise_matches_preds has null values'

        return finetuning_edges, pairwise_matches_preds

    def get_new_pairwise_matches_preds(self, pairwise_matches_preds, pred_buffer):
        """
        Adds to the pairwise_matches_preds the new predictions of the latest finetuned model
        """
        # First load the prediction log of the latest finetuned model

        new_pairwise_matches_preds = pd.DataFrame(pred_buffer)
        new_pairwise_matches_preds = new_pairwise_matches_preds.rename(columns={'lids': 'lid', 'rids': 'rid'})
        new_pairwise_matches_preds = self.sort_lids_rids(new_pairwise_matches_preds)
        new_pairwise_matches_preds = new_pairwise_matches_preds.rename(columns={'prediction_proba': 'finetune_prob'})

        # Add the new predictions to the pairwise_matches_preds
        pairwise_matches_preds = pairwise_matches_preds.merge(new_pairwise_matches_preds[['lid', 'rid', 'finetune_prob']], on=['lid', 'rid'], how='left')

        # If any pair has a 'manual_check_label' column different than -1, we set the prob column to the manual_check_label
        checked_indices = pairwise_matches_preds[pairwise_matches_preds['manual_check_label'] != -1].index
        pairwise_matches_preds.loc[checked_indices, 'prob'] = pairwise_matches_preds.loc[checked_indices, 'manual_check_label']

        return pairwise_matches_preds

        
    def update_pairwise_matches_preds(self, pairwise_matches_preds):
        """
        Updates the pairwise_matches_preds with the new predictions of the latest finetuned model using the following heuristic:

        1) We remove from the matching the minimum edge cuts of all subgraphs with one or more negative transitive predictions

        This heuristic naturally decreases the number of negative transitive predictions, as it breaks up the subgraphs into smaller subgraphs (which have less transitive matches).
        """
        pairwise_matches_preds['pairwise_id'] = pairwise_matches_preds.index # Add a pairwise_id column to the pairwise_matches_preds in order to ease the manipulation of the dataframe

        # Get the current true positives and false positives
        current_true_positives = pairwise_matches_preds[(pairwise_matches_preds['label'] == 1) & (pairwise_matches_preds['prob'] >= self.threshold)].shape[0]
        current_false_positives = pairwise_matches_preds[(pairwise_matches_preds['label'] == 0) & (pairwise_matches_preds['prob'] >= self.threshold)].shape[0]
        self.finetune_cleanup_dict['true_positives'].append(current_true_positives)
        self.finetune_cleanup_dict['false_positives'].append(current_false_positives)

        # Get the indices of the pairs that have not been manually checked
        no_manual_check_indices = pairwise_matches_preds[pairwise_matches_preds['manual_check_label'] == -1].index

        # Get the current matches_graph
        self.matches_graph = full_data_utils.generate_matches_graph(pairwise_matches_preds, threshold=self.threshold)
        subgraphs = list(nx.connected_components(self.matches_graph))
        subgraphs = [self.matches_graph.subgraph(c) for c in subgraphs]
        self.transitive_matches_preds = self.evaluate_transitive_edges(self.matches_graph)
        self.finetune_cleanup_dict['negative_transitive_preds'].append(self.transitive_matches_preds[self.transitive_matches_preds['prob'] < self.threshold].shape[0])
        self.finetune_cleanup_dict['positive_transitive_preds'].append(self.transitive_matches_preds[self.transitive_matches_preds['prob'] >= self.threshold].shape[0])


        # We add the transitive_preds to each subgraph
        subgraphs_with_transitive_preds = []
        for subgraph in subgraphs:
            subgraph_nodes = set(subgraph.nodes())
            subgraph_transitive_preds = self.transitive_matches_preds[self.transitive_matches_preds['lid'].isin(subgraph_nodes) & self.transitive_matches_preds['rid'].isin(subgraph_nodes)]
            if len(subgraph_transitive_preds) > 0:
                subgraphs_with_transitive_preds.append((subgraph, subgraph_transitive_preds))


        removed_true_positives_transitive = 0
        removed_false_positives_transitive = 0
        initial_false_positives = pairwise_matches_preds[(pairwise_matches_preds['label'] == 0) & (pairwise_matches_preds['prob'] >= self.threshold)].shape[0]
        initial_true_positives = pairwise_matches_preds[(pairwise_matches_preds['label'] == 1) & (pairwise_matches_preds['prob'] >= self.threshold)].shape[0]
        for subgraph, transitive_preds in tqdm(subgraphs_with_transitive_preds, total=len(subgraphs_with_transitive_preds), desc='[Transitive Preds Finetuning Cleanup Heuristic]'):
            neg_transitive_preds = transitive_preds[transitive_preds['prob'] < self.threshold]
            subgraph_edges = pd.DataFrame(subgraph.edges(data=True), columns=['lid', 'rid', 'match_type'])
            subgraph_edges = pd.concat([subgraph_edges.drop(['match_type'], axis=1), subgraph_edges['match_type'].apply(pd.Series)], axis=1)
            assert (subgraph_edges['prob'] >= self.threshold).all(), 'For an edge to be in the subgraph, it must have higher prob than the threshold'

            if len(neg_transitive_preds) > 0:
                min_edge_cut = nx.minimum_edge_cut(subgraph)
                min_edge_cut = pd.DataFrame(min_edge_cut, columns=['lid', 'rid'])
                min_edge_cut = self.sort_lids_rids(min_edge_cut)
                min_edge_cut = min_edge_cut.merge(pairwise_matches_preds, on=['lid', 'rid'], how='left')
                min_edge_cut_indices = min_edge_cut['pairwise_id']
                # We only want to update the predictions of the pairs that have not been manually checked, so we remove the indices of the pairs that have been manually checked
                min_edge_cut_indices = pd.Series(list(set(min_edge_cut_indices) & set(no_manual_check_indices)))
                removed_true_positives = pairwise_matches_preds.loc[min_edge_cut_indices, 'label'].sum()
                removed_false_positives = len(min_edge_cut_indices) - removed_true_positives
                removed_true_positives_transitive += removed_true_positives
                removed_false_positives_transitive += removed_false_positives
                assert (pairwise_matches_preds.loc[min_edge_cut_indices, 'prob'] >= self.threshold).all(), 'The prob of the pairs in the minimum edge cut must be higher than the threshold'
                pairwise_matches_preds.loc[min_edge_cut_indices, 'prob'] = 0

                assert  pairwise_matches_preds[(pairwise_matches_preds['label'] == 0) & (pairwise_matches_preds['prob'] >= self.threshold)].shape[0] == initial_false_positives - removed_false_positives_transitive, 'The number of false positives in the pairwise_matches_preds is not correct'
                assert  pairwise_matches_preds[(pairwise_matches_preds['label'] == 1) & (pairwise_matches_preds['prob'] >= self.threshold)].shape[0] == initial_true_positives - removed_true_positives_transitive, 'The number of true positives in the pairwise_matches_preds is not correct'

        self.finetune_cleanup_dict['removed_true_positives_transitive'].append(removed_true_positives_transitive)
        self.finetune_cleanup_dict['removed_false_positives_transitive'].append(removed_false_positives_transitive)
        self.finetune_cleanup_dict['labeled_pairs'].append(len(self.pairwise_matches_preds[self.pairwise_matches_preds['manual_check_label'] != -1]))

        # If any pair has a 'manual_check_label' column different than -1, we set the prob column to the manual_check_label
        checked_indices = pairwise_matches_preds[pairwise_matches_preds['manual_check_label'] != -1].index
        pairwise_matches_preds.loc[checked_indices, 'prob'] = pairwise_matches_preds.loc[checked_indices, 'manual_check_label']

        pairwise_matches_preds = pairwise_matches_preds.drop(columns=['pairwise_id'])

        return pairwise_matches_preds

    
    def finetune_model(self, finetuning_edges, finetune_results_path, finetuning_iteration, test = False):
        """
        Finetunes the model with the finetuning_edges for as many epochs as specified in the args.

        """

        self.model.train_data_loader.dataset.idx_df = finetuning_edges[['lid', 'rid', 'label']]
        self.model.test_data_loader.dataset.idx_df = self.pairwise_matches_preds[self.pairwise_matches_preds['prob'] >= self.threshold][['lid', 'rid', 'label']]

        # Finetune the model 
        self.model.use_val = False
        self.model.fine_tune(epochs=self.args.finetuning_epochs, finetune_results_path=finetune_results_path, finetuning_iteration=finetuning_iteration, test_after_finetune=test)
        if hasattr(self.model, 'finetune_dict'):
            # Save the finetune dict
            finetune_dict_df = pd.DataFrame(self.model.finetune_dict)
            finetune_dict_df.to_csv(os.path.join(finetune_results_path, 'finetune_dict.csv'), index=False)

    def finetune_CLER_model(self, finetuning_edges):
        """
        Finetunes the CLER model with the finetuning_edges for as many epochs as specified in the args.

        """
        finetune_df = copy.deepcopy(finetuning_edges)
        attr_listA, entity_listA, _, _ = load_attributes(dataset_raw_file_path(Config.DATASETS[self.model.dataset.name]), load_test=True)
        finetune_df, id2idx = translate_ids(dataset_raw_file_path(Config.DATASETS[self.model.dataset.name]), finetune_df, load_test=True)
        # Set the finetuning edges as a list of (lid, rid, label) tuples
        finetune_df = finetune_df[['lid', 'rid', 'label']]
        finetune_df = [tuple(x) for x in finetune_df.to_numpy()]
        finetune_df = np.concatenate([finetune_df, [[1]]*len(finetune_df)], 1) # 1 weights for pseudo labels
        # Finetune the model
        train_set = GTDatasetWithLabelWeights(finetune_df, entity_listA, entity_listA, attr_listA, lm= self.model.lm, concat=True, shuffle=False)
        train_loss = train_matcher(self.model, train_set, self.model.optimizer, self.args, scheduler=None)
        # Save the model after finetuning
        torch.save(self.model.state_dict(), os.path.join(self.finetune_results_path, 'CLER_finetune_iteration_{}.pt'.format(self.finetuning_iteration)))

    def evaluate_edges(self, edges_df, leave_tqdm=True):
        """
        Evaluates the edges included in the edges_df with the corresponding model.
        """
        if "CLER" in self.model.args.model_name:
            return self.evaluate_edges_CLER(edges_df)
        else:
            return self.evaluate_edges_pairwise(edges_df, leave_tqdm=leave_tqdm)

    def evaluate_edges_pairwise(self, edges_df, leave_tqdm=True):
        """
        Evaluates the edges_df with the pairwise matching model

        Outputs:

        - predictions: DataFrame with the columns ['lid', 'rid', 'prob'] containing the predictions of the model for the edges_df
        """

        # Create a new dataloader with the edges_df
        eval_data_loader = copy.deepcopy(self.model.test_data_loader)
        eval_data_loader.dataset.idx_df = edges_df[['lid', 'rid', 'label']]

        # Evaluate the model
        if leave_tqdm:
            print('Evaluating {} transitive matches'.format(len(edges_df)))
        predictions = self.model.evaluate_dataloader(eval_data_loader, leave_tqdm=leave_tqdm)
        predictions =  pd.DataFrame(predictions)
        predictions = predictions.rename(columns={'lids': 'lid', 'rids': 'rid', 'prediction_proba': 'prob'})
        predictions = self.sort_lids_rids(predictions)
        return predictions
    
    def evaluate_edges_CLER(self, edges_df):
        """
        Evaluates the edges_df with the CLER pairwise matching model

        Outputs:

        - predictions: DataFrame with the columns ['lid', 'rid', 'prob'] containing the predictions of the model for the edges_df
        """
        attr_listA, entity_listA, _, _ = load_attributes(dataset_raw_file_path(Config.DATASETS[self.model.dataset.name]), load_test=True)
        edges_df, id2idx = translate_ids(dataset_raw_file_path(Config.DATASETS[self.model.dataset.name]), edges_df, load_test=True)
        edges_df = edges_df[['lid', 'rid', 'label']]
        edges = [tuple(x) for x in edges_df.to_numpy()]

        batch_size = 64
        test_set = GTDatasetWithLabel(edges, entity_listA, entity_listA, attr_listA, lm= self.model.lm, concat=True, shuffle=False)
        self.model.eval()
        iterator = DataLoader(dataset= test_set, batch_size= batch_size, collate_fn=test_set.pad)
        preds = []

        for i, batch in tqdm(enumerate(iterator), total=len(iterator), leave= False):
            batch_pred, _ = self.pred_batch(batch)
            preds.append(batch_pred)
        
        preds = np.concatenate(preds)
        edges_df['prob'] = preds
        edges_df.index = zip(edges_df['lid'], edges_df['rid'])
        edges_df = translate_ids_back(edges_df, id2idx)
        return edges_df[['lid', 'rid', 'prob']]

    def pred_batch(self, batch):
        x, _, y = batch
        x = x.cuda()
        logits = self.model(x)
        scores = logits.argmax(-1)
        y_pre = scores.cpu().numpy().tolist()
        y_scores = logits.softmax(-1)[:,1].cpu().detach().numpy().tolist()
        return np.array(y_pre), np.array(y_scores)


    
    def evaluate_transitive_edges(self, matches_graph):
        """
        Evaluates the transitive matches of the current matches_graph with the pairwise matching model
        """

        transitive_matches = full_data_utils.get_transitive_matches_finetuning(matches_graph, subgraph_size_threshold=self.subgraph_size_threshold)

        transitive_matches = pd.DataFrame(transitive_matches, columns=['lid', 'rid', 'match_type'])
        transitive_matches = self.sort_lids_rids(transitive_matches)
        # We add a dummy label to the transitive matches (we only want to evaluate them w/ the model)
        transitive_matches['label'] = 0
        # Now we need to evaluate the transitive matches with the pairwise matching model
        transitive_matches = transitive_matches.drop_duplicates(subset=['lid', 'rid']) # We only need to evaluate each pair once
        transitive_matches_preds = self.evaluate_edges(transitive_matches)
        return transitive_matches_preds

    def filter_pairwise_matches_preds(self, pairwise_matches_preds, threshold):
        """
        Initial filter of the pairwise_matches_preds to remove the pairs with a prediction below the threshold and duplicates
        """
        # Remove the pairs with a prediction below the threshold
        pairwise_matches_preds = pairwise_matches_preds[pairwise_matches_preds['prob'] >= threshold]

        # Drop duplicates
        pairwise_matches_preds = pairwise_matches_preds.drop_duplicates(subset=['lid', 'rid'])

        return pairwise_matches_preds

    def check_results_folder(self, finetuning_iteration):
        """
        Check whether the results folder of a given epoch exists and has all the output files.
        """
        results_path = self.finetune_results_path
        if not os.path.exists(results_path):
            return False
        elif 'CLER' in self.model.args.model_name:
            if os.path.exists(os.path.join(results_path, 'CLER_finetune_iteration_{}.pt'.format(finetuning_iteration))) and \
               os.path.exists(os.path.join(results_path, 'pairwise_matches_preds.csv')) and os.path.exists(os.path.join(results_path, 'finetuning_edges.csv')) and \
               os.path.exists(os.path.join(results_path, 'finetune_cleanup_dict.csv')):
                
                return True
            else:
                return False
        else:
            if os.path.exists(os.path.join(results_path, '{}_finetune_iteration_{}.pt'.format(self.model.args.model_name, finetuning_iteration))) and \
               os.path.exists(os.path.join(results_path, 'pairwise_matches_preds.csv')) and os.path.exists(os.path.join(results_path, 'finetuning_edges.csv')) and \
               os.path.exists(os.path.join(results_path, 'finetune_cleanup_dict.csv')):
                
                return True
            else:
                return False
            
    #######################################################################

    # Post-finetuning Cleanup

    #######################################################################

    def post_finetuning_cleanup(self):
        """
        Perform the post-finetuning cleanup of the pairwise_matches_preds. We remove the minimum edge cut pairs of all subgraphs with more negative transitive predictions than positive ones.
        """
        print('##############################################')
        print('Starting post-finetuning cleanup')
        print('##############################################')
        self.post_finetuning_cleanup_dict = {'starting_true_positives': [], 'starting_false_positives': [],
                                             'starting_negative_transitive_preds': [], 'starting_positive_transitive_preds': [],
                                             'removed_false_positives_cleanup': [], 'removed_true_positives_cleanup': []}
        
        self.post_finetuning_cleanup_dict['starting_true_positives'].append(len(self.pairwise_matches_preds[(self.pairwise_matches_preds['label'] == 1) & (self.pairwise_matches_preds['prob'] >= self.threshold)]))
        self.post_finetuning_cleanup_dict['starting_false_positives'].append(len(self.pairwise_matches_preds[(self.pairwise_matches_preds['label'] == 0) & (self.pairwise_matches_preds['prob'] >= self.threshold)]))
        matches_graph = full_data_utils.generate_matches_graph(self.pairwise_matches_preds, threshold=self.threshold)

        if hasattr(self, 'transitive_matches_preds'):
            transitive_matches_preds = self.transitive_matches_preds
        else:
            transitive_matches_preds = self.evaluate_transitive_edges(matches_graph)

        self.post_finetuning_cleanup_dict['starting_negative_transitive_preds'].append(len(transitive_matches_preds[transitive_matches_preds['prob'] < self.threshold]))
        self.post_finetuning_cleanup_dict['starting_positive_transitive_preds'].append(len(transitive_matches_preds[transitive_matches_preds['prob'] >= self.threshold]))

        self.pairwise_matches_preds['pairwise_id'] = self.pairwise_matches_preds.index
        unchecked_indices = self.pairwise_matches_preds[self.pairwise_matches_preds['manual_check_label'] == -1].index
        matches_graph = full_data_utils.generate_matches_graph(self.pairwise_matches_preds, threshold=self.threshold)
        subgraphs_with_transitive_preds = self.get_subgraphs_with_transitive_preds(matches_graph, transitive_matches_preds)
        
        # We now iterate over the subgraphs and remove the minimum edge cuts of the subgraphs with more negative transitive predictions than positive ones
        stop_cleanup = False
        removed_true_positives_cleanup = 0
        removed_false_positives_cleanup = 0

        while not stop_cleanup:
            update_subgraphs = False
            for subgraph, transitive_preds in subgraphs_with_transitive_preds:
                neg_transitive_preds = transitive_preds[transitive_preds['prob'] < self.threshold]
                pos_transitive_preds = transitive_preds[transitive_preds['prob'] >= self.threshold]
                if len(neg_transitive_preds) > len(pos_transitive_preds):
                    min_edge_cut = nx.minimum_edge_cut(subgraph)
                    min_edge_cut = pd.DataFrame(min_edge_cut, columns=['lid', 'rid'])
                    min_edge_cut = self.sort_lids_rids(min_edge_cut)
                    min_edge_cut = min_edge_cut.merge(self.pairwise_matches_preds, on=['lid', 'rid'], how='left')
                    min_edge_cut_indices = min_edge_cut['pairwise_id']
                    # We only want to update the predictions of the pairs that have not been manually checked, so we remove the indices of the pairs that have been manually checked
                    min_edge_cut_indices = pd.Series(list(set(min_edge_cut_indices) & set(unchecked_indices)))
                    self.pairwise_matches_preds.loc[min_edge_cut_indices, 'prob'] = 0
                    removed_true_positives = self.pairwise_matches_preds.loc[min_edge_cut_indices, 'label'].sum()
                    removed_false_positives = len(min_edge_cut_indices) - removed_true_positives

                    removed_true_positives_cleanup += removed_true_positives
                    removed_false_positives_cleanup += removed_false_positives

                    if len(min_edge_cut_indices) > 0:
                        update_subgraphs = True
                else:
                    # Remove the subgraph from the subgraphs_with_transitive_preds
                    subgraphs_with_transitive_preds.pop(subgraphs_with_transitive_preds.index((subgraph, transitive_preds)))

            if update_subgraphs:
                # Refresh the subgraphs
                matches_graph = full_data_utils.generate_matches_graph(self.pairwise_matches_preds, threshold=self.threshold)
                subgraphs_with_transitive_preds = self.get_subgraphs_with_transitive_preds(matches_graph, transitive_matches_preds)

            # Check if subgraphs_with_transitive_preds is empty
            if len(subgraphs_with_transitive_preds) == 0 or update_subgraphs == False:
                stop_cleanup = True

        self.pairwise_matches_preds = self.pairwise_matches_preds.drop(columns=['pairwise_id'])
        self.post_finetuning_cleanup_dict['removed_true_positives_cleanup'].append(removed_true_positives_cleanup)
        self.post_finetuning_cleanup_dict['removed_false_positives_cleanup'].append(removed_false_positives_cleanup)

        # Save the post-finetuning cleanup dict
        post_finetuning_cleanup_path = os.path.join(self.post_finetuning_results_path, 'post_finetuning_cleanup')
        if not os.path.exists(post_finetuning_cleanup_path):
            os.makedirs(post_finetuning_cleanup_path)

        post_finetuning_cleanup_df = pd.DataFrame(self.post_finetuning_cleanup_dict)
        post_finetuning_cleanup_df.to_csv(os.path.join(post_finetuning_cleanup_path, 'post_finetuning_cleanup.csv'), index=False)
        
        return self.pairwise_matches_preds

    def get_subgraphs_with_transitive_preds(self, matches_graph, transitive_matches_preds):

        subgraphs = list(nx.connected_components(matches_graph))
        subgraphs = [matches_graph.subgraph(c) for c in subgraphs]

        # We add the transitive_preds to each subgraph
        subgraphs_with_transitive_preds = []
        for subgraph in subgraphs:
            subgraph_nodes = set(subgraph.nodes())
            subgraph_transitive_preds = transitive_matches_preds[transitive_matches_preds['lid'].isin(subgraph_nodes) & transitive_matches_preds['rid'].isin(subgraph_nodes)]
            if len(subgraph_transitive_preds) > 0:
                subgraphs_with_transitive_preds.append((subgraph, subgraph_transitive_preds))
        
        return subgraphs_with_transitive_preds
    
    ###########################################################################

    # Post-Finetuning Checks

    ###########################################################################

    def post_finetuning_checks(self):
        """
        We perform post-finetuning checks:
        - 1) Pre-check for the very large subgraphs.
        - 2) If any of the transitive matches has been checked as a false positive, we manually check all edges of the subgraph it belongs to in the pairwise_matches_preds.
        - 3) We evaluate all the transitive matches with the final fine-tuned model, if any of them is predicted as a non-match, we manually check all the edges of the subgraph it belongs to.
        """
        self.post_finetuning_checks_dict = {'starting_negative_transitive_preds': [], 'starting_positive_transitive_preds': [],
                                'starting_true_positives': [], 'starting_false_positives': [], 'removed_true_positives': [],
                                'removed_false_positives': [], 'labeled_pairs_large_subgraphs_check':[],
                                'labeled_pairs_full_subgraph_check_heuristic':[], 'labeled_pairs_transitive_check_heuristic':[],}

        self.post_finetuning_checks_dict['starting_true_positives'].append(len(self.pairwise_matches_preds[(self.pairwise_matches_preds['label'] == 1) & (self.pairwise_matches_preds['prob'] >= self.threshold)]))
        self.post_finetuning_checks_dict['starting_false_positives'].append(len(self.pairwise_matches_preds[(self.pairwise_matches_preds['label'] == 0) & (self.pairwise_matches_preds['prob'] >= self.threshold)]))

        print('##############################################')
        print('Starting post-finetuning checks')
        print('##############################################')

        total_removed_false_positives = 0
        total_removed_true_positives = 0
        post_finetuning_effort = 0
        # First, we do a pre-check for the very large subgraphs

        matches_graph = full_data_utils.generate_matches_graph(self.pairwise_matches_preds, threshold=self.threshold)
        self.pairwise_matches_preds, manual_check_effort, removed_false_positives, removed_true_positives = self.large_subgraphs_check(matches_graph, subgraph_size_threshold=self.subgraph_size_threshold)
        post_finetuning_effort += manual_check_effort
        total_removed_false_positives += removed_false_positives
        total_removed_true_positives += removed_true_positives

        # Second, we generate the transitive matches of the post-finetuning matches_graph
        matches_graph = full_data_utils.generate_matches_graph(self.pairwise_matches_preds, threshold=self.threshold)
        transitive_matches = full_data_utils.get_all_transitive_matches(matches_graph)

        matches_graph_pre_check = full_data_utils.add_transitive_edges_to_graph(matches_graph)
        # If any of the transitive matches has been checked as a false positive, we manually check all edges of the subgraph it belongs to in the pairwise_matches_preds
        self.pairwise_matches_preds, manual_check_effort, removed_false_positives, removed_true_positives  = self.full_subgraph_check_heuristic(matches_graph_pre_check, transitive_matches)
        post_finetuning_effort += manual_check_effort
        total_removed_false_positives += removed_false_positives
        total_removed_true_positives += removed_true_positives

        # We evaluate all the transitive matches with the final fine-tuned model, if any of them is predicted as a non-match, we manually check all the edges of the subgraph it belongs to
        self.pairwise_matches_preds, manual_check_effort, removed_false_positives, removed_true_positives = self.transitive_check_heuristic(matches_graph_pre_check, transitive_matches)
        post_finetuning_effort += manual_check_effort
        total_removed_false_positives += removed_false_positives
        total_removed_true_positives += removed_true_positives

        self.post_finetuning_checks_dict['removed_true_positives'].append(total_removed_true_positives)
        self.post_finetuning_checks_dict['removed_false_positives'].append(total_removed_false_positives)
        self.remaining_labeling_budget = self.args.labeling_budget - len(self.finetuning_edges) - post_finetuning_effort

        # Save the post-finetuning checks dict
        post_finetuning_checks_path = os.path.join(self.post_finetuning_results_path, 'post_finetuning_check')
        if not os.path.exists(post_finetuning_checks_path):
            os.makedirs(post_finetuning_checks_path)

        post_finetuning_checks_df = pd.DataFrame(self.post_finetuning_checks_dict)
        post_finetuning_checks_df.to_csv(os.path.join(post_finetuning_checks_path, 'post_finetuning_check.csv'), index=False)

        return post_finetuning_effort


    def full_subgraph_check_heuristic(self, matches_graph, transitive_edges):
        """
        If any of the transitive matches has been checked as a false positive, we manually check all edges of the subgraph it belongs to (breaking it up)
        """
        self.pairwise_matches_preds['pairwise_id'] = self.pairwise_matches_preds.index

        checked_false_positives = self.pairwise_matches_preds[self.pairwise_matches_preds['manual_check_label'] == 0]
        checked_false_positives = self.sort_lids_rids(checked_false_positives)
        checked_false_positives_set = set(zip(checked_false_positives['lid'], checked_false_positives['rid']))

        manual_check_effort = 0
        removed_false_positives = 0
        removed_true_positives = 0
        for transitive_edge in transitive_edges:
            # Check if the transitive edge has been checked as a false positive
            if (transitive_edge[0], transitive_edge[1]) in checked_false_positives_set or (transitive_edge[1], transitive_edge[0]) in checked_false_positives_set:
                # Get the subgraph that contains the transitive edge
                subgraph = matches_graph.subgraph(nx.node_connected_component(matches_graph, transitive_edge[0]))
                subgraph_edges = pd.DataFrame(subgraph.edges(data=True), columns=['lid', 'rid', 'match_type'])
                subgraph_edges = pd.concat([subgraph_edges.drop(['match_type'], axis=1), subgraph_edges['match_type'].apply(pd.Series)], axis=1)
                subgraph_edges = subgraph_edges[subgraph_edges['match_type'] != 'transitive_match']
                subgraph_edges = self.sort_lids_rids(subgraph_edges)
                # We modify the prob and manual_check_label columns of each of the subgraph edges in the pairwise_matches_preds
                subgraph_edges = subgraph_edges[['lid', 'rid']]
                # Get the indices of the subgraph_edges in the pairwise_matches_preds
                subgraph_edges = self.pairwise_matches_preds.merge(subgraph_edges, on=['lid', 'rid'], how='inner')
                unchecked_subgraph_edges = subgraph_edges[subgraph_edges['manual_check_label'] == -1]
                if self.args.manual_check:
                    unchecked_subgraph_edges = unchecked_subgraph_edges['pairwise_id']
                    removed_false_positives += len(subgraph_edges[(subgraph_edges['label'] == 0) & (subgraph_edges['manual_check_label'] == -1)])
                    # Set the manual_check_label to be the same as the label
                    self.pairwise_matches_preds.loc[unchecked_subgraph_edges, 'manual_check_label'] = self.pairwise_matches_preds.loc[unchecked_subgraph_edges, 'label']
                    # Set the prob to be the same as the manual_check_label
                    self.pairwise_matches_preds.loc[unchecked_subgraph_edges, 'prob'] = self.pairwise_matches_preds.loc[unchecked_subgraph_edges, 'manual_check_label'] 
                    manual_check_effort += len(unchecked_subgraph_edges)
                else:
                    # We label the unchecked edges with the LLM
                    self.pairwise_matches_preds, removed_false_positives_LLM, removed_true_positives_LLM = self.LLM_labeling_post_finetuning(unchecked_edges=unchecked_subgraph_edges)
                    removed_false_positives += removed_false_positives_LLM
                    removed_true_positives += removed_true_positives_LLM

        self.pairwise_matches_preds = self.pairwise_matches_preds.drop(columns=['pairwise_id'])
        self.post_finetuning_checks_dict['labeled_pairs_full_subgraph_check_heuristic'].append(manual_check_effort)

        return self.pairwise_matches_preds, manual_check_effort, removed_false_positives, removed_true_positives
    
    def transitive_check_heuristic(self, matches_graph, transitive_matches):
        """
        We evaluate all the transitive matches with the final fine-tuned model, if any of them is predicted as a non-match, we manually check all the edges of the subgraph it belongs to
        """
        self.pairwise_matches_preds['pairwise_id'] = self.pairwise_matches_preds.index

        if len(transitive_matches) == 0:
            self.post_finetuning_checks_dict['starting_negative_transitive_preds'].append(0)
            self.post_finetuning_checks_dict['starting_positive_transitive_preds'].append(0)
            self.post_finetuning_checks_dict['labeled_pairs_transitive_check_heuristic'].append(0)
            return self.pairwise_matches_preds, 0 , 0, 0

        transitive_matches = pd.DataFrame(transitive_matches, columns=['lid', 'rid', 'match_type'])
        transitive_matches = self.sort_lids_rids(transitive_matches).drop_duplicates(subset=['lid', 'rid'])
        # We add a dummy label to the transitive edges (we only want to evaluate them w/ the model)
        transitive_matches['label'] = 0
        # Now we need to evaluate the transitive matches with the pairwise matching model
        transitive_matches_preds = self.evaluate_edges(transitive_matches)
        self.post_finetuning_checks_dict['starting_negative_transitive_preds'].append(len(transitive_matches_preds[transitive_matches_preds['prob'] < self.threshold]))
        self.post_finetuning_checks_dict['starting_positive_transitive_preds'].append(len(transitive_matches_preds[transitive_matches_preds['prob'] >= self.threshold]))

        subgraphs = list(nx.connected_components(matches_graph))
        subgraphs = [matches_graph.subgraph(c) for c in subgraphs]
        manual_check_effort = 0
        removed_false_positives = 0
        removed_true_positives = 0

        for subgraph in tqdm(subgraphs, total=len(subgraphs), desc='[Transitive Check Heuristic]'):
            subgraph_edges = pd.DataFrame(subgraph.edges(data=True), columns=['lid', 'rid', 'match_type'])
            subgraph_edges = pd.concat([subgraph_edges.drop(['match_type'], axis=1), subgraph_edges['match_type'].apply(pd.Series)], axis=1)
            transitive_edges = copy.deepcopy(subgraph_edges[subgraph_edges['match_type'] == 'transitive_match'])
            if transitive_edges.empty:
                continue
            transitive_edges = self.sort_lids_rids(transitive_edges)
            transitive_edges = transitive_edges.drop(columns=['prob'])
            transitive_edges = transitive_edges.merge(transitive_matches_preds[['lid', 'rid', 'prob']], on=['lid', 'rid'], how='left')

            # Check if any of the transitive edges are being predicted as non-matches
            transitive_edges['neg_pred'] = transitive_edges['prob'] < self.threshold
            negative_transitive_preds = transitive_edges['neg_pred'].sum() / len(transitive_edges)

            if negative_transitive_preds > 0: 
                # This subgraph has a transitive edge that is being predicted as a non-match, we manually check all the edges of the subgraph
                subgraph_edges = subgraph_edges[subgraph_edges['match_type'] != 'transitive_match']
                subgraph_edges = subgraph_edges.drop(columns=['prob', 'match_type'])
                subgraph_edges = self.sort_lids_rids(subgraph_edges)
                # We modify the prob and manual_check_label columns of each of the subgraph edges in the pairwise_matches_preds
                subgraph_edges = subgraph_edges[['lid', 'rid']]
                # Get the indices of the subgraph_edges in the pairwise_matches_preds
                subgraph_edges = self.pairwise_matches_preds.merge(subgraph_edges, on=['lid', 'rid'], how='inner')
                unchecked_subgraph_edges = subgraph_edges[subgraph_edges['manual_check_label'] == -1]
                if self.args.manual_check:
                    unchecked_subgraph_edges = unchecked_subgraph_edges['pairwise_id']
                    removed_false_positives += len(subgraph_edges[(subgraph_edges['label'] == 0) & (subgraph_edges['manual_check_label'] == -1)])
                    # Set the manual_check_label to be the same as the label
                    self.pairwise_matches_preds.loc[unchecked_subgraph_edges, 'manual_check_label'] = self.pairwise_matches_preds.loc[unchecked_subgraph_edges, 'label']
                    # Set the prob to be the same as the manual_check_label
                    self.pairwise_matches_preds.loc[unchecked_subgraph_edges, 'prob'] = self.pairwise_matches_preds.loc[unchecked_subgraph_edges, 'manual_check_label']
                    manual_check_effort += len(unchecked_subgraph_edges)
                else:
                    # We label the unchecked edges with the LLM
                    self.pairwise_matches_preds, removed_false_positives_LLM, removed_true_positives_LLM = self.LLM_labeling_post_finetuning(unchecked_edges=unchecked_subgraph_edges)
                    removed_false_positives += removed_false_positives_LLM
                    removed_true_positives += removed_true_positives_LLM
        
        self.pairwise_matches_preds = self.pairwise_matches_preds.drop(columns=['pairwise_id'])
        self.post_finetuning_checks_dict['labeled_pairs_transitive_check_heuristic'].append(manual_check_effort)

        return self.pairwise_matches_preds, manual_check_effort, removed_false_positives, removed_true_positives
    
    def large_subgraphs_check(self, matches_graph, subgraph_size_threshold=50):
        """
        We check whether any of the subgraphs in the matches_graph is too large (i.e. has a size greater than subgraph_size_threshold). If so, we manually check all the edges of the subgraph.
        """
        self.pairwise_matches_preds['pairwise_id'] = self.pairwise_matches_preds.index

        subgraphs = list(nx.connected_components(matches_graph))
        subgraphs = [matches_graph.subgraph(c) for c in subgraphs]

        manual_check_effort = 0
        removed_false_positives = 0
        removed_true_positives = 0

        for subgraph in subgraphs:
            if len(subgraph) > subgraph_size_threshold:
                subgraph_edges = pd.DataFrame(subgraph.edges(data=True), columns=['lid', 'rid', 'match_type'])
                subgraph_edges = pd.concat([subgraph_edges.drop(['match_type'], axis=1), subgraph_edges['match_type'].apply(pd.Series)], axis=1)
                subgraph_edges = self.sort_lids_rids(subgraph_edges)
                subgraph_edges = subgraph_edges[['lid', 'rid']]
                # Get the indices of the subgraph_edges in the pairwise_matches_preds
                subgraph_edges = self.pairwise_matches_preds.merge(subgraph_edges, on=['lid', 'rid'], how='inner')
                
                if self.args.manual_check:
                    unchecked_edges = subgraph_edges[subgraph_edges['manual_check_label'] == -1]['pairwise_id']
                    removed_false_positives += len(subgraph_edges[(subgraph_edges['label'] == 0) & (subgraph_edges['manual_check_label'] == -1)])
                    # Set the manual_check_label to be the same as the label
                    self.pairwise_matches_preds.loc[unchecked_edges, 'manual_check_label'] = self.pairwise_matches_preds.loc[unchecked_edges, 'label']
                    # Set the prob to be the same as the manual_check_label
                    self.pairwise_matches_preds.loc[unchecked_edges, 'prob'] = self.pairwise_matches_preds.loc[unchecked_edges, 'manual_check_label']
                    manual_check_effort += len(unchecked_edges)
                else:
                    # We label the edges of the subgraph via the LLM
                    self.pairwise_matches_preds, removed_false_positives_LLM, removed_true_positives_LLM = self.LLM_labeling_post_finetuning(subgraph_edges[subgraph_edges['manual_check_label'] == -1])
                    removed_false_positives += removed_false_positives_LLM
                    removed_true_positives += removed_true_positives_LLM

        self.pairwise_matches_preds = self.pairwise_matches_preds.drop(columns=['pairwise_id'])
        self.post_finetuning_checks_dict['labeled_pairs_large_subgraphs_check'].append(manual_check_effort)

        return self.pairwise_matches_preds, manual_check_effort, removed_false_positives, removed_true_positives

    def LLM_labeling_post_finetuning(self, unchecked_edges):
        """
        We label the unchecked edges of a subgraph via the LLM, adding the labels to the pairwise_matches_preds dataframe.
        """
        # Load the LLM model
        self.load_LLM_model()

        removed_false_positives = 0
        removed_true_positives = 0

        for i, edge in tqdm(unchecked_edges.iterrows(), desc='[LLM Labeling]', total=len(unchecked_edges)):
            lid_record = self.model.dataset.tokenized_data.loc[edge['lid']]['concat_data']
            rid_record = self.model.dataset.tokenized_data.loc[edge['rid']]['concat_data']

            # We use the LLM to label the edge
            self.pairwise_matches_preds.loc[edge['pairwise_id'], 'manual_check_label'] = label_record_pair_with_LLM(self.LLM_model, self.LLM_tokenizer, self.LLM_model_max_length, lid_record, rid_record)
            self.pairwise_matches_preds.loc[edge['pairwise_id'], 'prob'] = self.pairwise_matches_preds.loc[edge['pairwise_id'], 'manual_check_label']
            if self.pairwise_matches_preds.loc[edge['pairwise_id'], 'label'] == 1:
                removed_true_positives += 1
            else:
                removed_false_positives += 1

        # Delete the LLM model to free up memory
        del self.LLM_model

        return self.pairwise_matches_preds, removed_false_positives, removed_true_positives


            
    ###########################################################################

    # Edge Recovery Functions

    ###########################################################################

    def edge_recovery(self, labeling_budget):
        """
        Perform the edge recovery process to recover the true positives that have been removed during the finetuning process.

        For every deleted edge, we evaluate  the transitive edges of the subgraph it would create if it was added back to the matches_graph.
        If all the transitive edges are predicted as matches, we add the edge back to the matches_graph. Otherwise, we manually check the edge (if there is labeling budget left).
        """
        self.pairwise_matches_preds['pairwise_id'] = self.pairwise_matches_preds.index
        print('##############################################')
        print('Starting edge recovery process')
        print('##############################################')
        pre_edge_recovery_true_positives = self.pairwise_matches_preds[(self.pairwise_matches_preds['prob'] >= self.threshold) & (self.pairwise_matches_preds['label'] == 1)].shape[0]
        pre_edge_recovery_false_positives = self.pairwise_matches_preds[(self.pairwise_matches_preds['prob'] >= self.threshold) & (self.pairwise_matches_preds['label'] == 0)].shape[0]

        deleted_edges = self.pairwise_matches_preds[(self.pairwise_matches_preds['prob'] == 0) & (self.pairwise_matches_preds['manual_check_label'] == -1)]
        if len(deleted_edges) == 0:
            print('No deleted edges to recover')
            self.save_edge_recovery_results(false_positives_readded = 0, true_positives_recovered = 0, labeling_budget = labeling_budget,
                                            pre_edge_recovery_true_positives = pre_edge_recovery_true_positives,
                                            pre_edge_recovery_false_positives = pre_edge_recovery_false_positives)
            return 
        deleted_edges_preds = self.evaluate_edges(copy.deepcopy(deleted_edges), leave_tqdm=False)
        deleted_edges = deleted_edges.drop(columns = 'prob')
        deleted_edges = deleted_edges.merge(deleted_edges_preds, on=['lid', 'rid'], how='left')
        deleted_edges = deleted_edges[deleted_edges['prob'] >= self.threshold]
        
        all_transitive_preds = self.get_all_transitive_preds(deleted_edges)
        true_positives_recovered = 0
        false_positives_readded = 0
        matches_graph = full_data_utils.generate_matches_graph(self.pairwise_matches_preds, threshold=self.threshold)
        subgraphs = list(nx.connected_components(matches_graph))
        subgraphs = [matches_graph.subgraph(c) for c in subgraphs]


        for i, deleted_edge in tqdm(deleted_edges.iterrows(), desc='[Edge Recovery]', total=len(deleted_edges)):
            lid = deleted_edge['lid']
            rid = deleted_edge['rid']
            # Get the subgraphs that contain the lid and rid records (they can also be the same subgraph)
            subgraph_with_lid = [subgraph for subgraph in subgraphs if lid in subgraph.nodes()]
            subgraph_with_rid = [subgraph for subgraph in subgraphs if rid in subgraph.nodes()]
            if len(subgraph_with_lid) == 0 and len(subgraph_with_rid) == 0:
                # The edge connects two records with no other matches in the matches_graph. Since we cannot evaluate any transitive edges, we manually check the edge
                if self.args.manual_check:
                    # We manually check the edge if the labeling budget is not exhausted
                    if labeling_budget > 0:
                        self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'manual_check_label'] = self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label']
                        self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'prob'] = self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label']
                        labeling_budget -= 1
                        if self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label'] == 1:
                            true_positives_recovered += 1
                            lid_rid_subgraph = nx.Graph([(lid, rid)])
                            subgraphs.append(lid_rid_subgraph) # Add the new subgraph to the list of subgraphs
                continue

            # Check whether the records are in the same subgraph
            try:
                lid_subgraph_nodes = set(subgraph_with_lid[0].nodes())
            except:
                lid_subgraph_nodes = set([lid])
            try:
                rid_subgraph_nodes = set(subgraph_with_rid[0].nodes())
            except:
                rid_subgraph_nodes = set([rid])

            if lid_subgraph_nodes == rid_subgraph_nodes:
                # If both records are in the same subgraph, the edge is already in the matches_graph, as it is a transitive edge. Consequently, we can ignore it here.
                continue
            else:
                # Both records are in different subgraphs. To evaluate the transitive edges, we create a new subgraph that contains all nodes of the previous subgraphs
                lid_rid_subgraph = nx.Graph([(lid, rid)])
                new_subgraph = self.get_new_subgraph(subgraph_with_lid, subgraph_with_rid, lid_rid_subgraph)
                if len(new_subgraph) > self.subgraph_size_threshold:
                    # The new subgraph is too large, this is likely due to multiple false positives being added back to the matches_graph. We manually check all the unchecked edges in the subgraph
                    subgraph_edges = pd.DataFrame(new_subgraph.edges(data=True), columns=['lid', 'rid', 'match_type'])
                    subgraph_edges = pd.concat([subgraph_edges.drop(['match_type'], axis=1), subgraph_edges['match_type'].apply(pd.Series)], axis=1)
                    subgraph_edges = self.sort_lids_rids(subgraph_edges)
                    subgraph_edges = subgraph_edges[['lid', 'rid']].merge(self.pairwise_matches_preds, on=['lid', 'rid'], how='left')
                    unchecked_edges = subgraph_edges[subgraph_edges['manual_check_label'] == -1]
                    if self.args.manual_check:
                        if len(unchecked_edges) < labeling_budget:
                        # We manually check all the edges of the subgraph
                            for i, unchecked_edge in unchecked_edges.iterrows():
                                self.pairwise_matches_preds.loc[unchecked_edge['pairwise_id'], 'manual_check_label'] = self.pairwise_matches_preds.loc[unchecked_edge['pairwise_id'], 'label']
                                self.pairwise_matches_preds.loc[unchecked_edge['pairwise_id'], 'prob'] = self.pairwise_matches_preds.loc[unchecked_edge['pairwise_id'], 'label']
                                if self.pairwise_matches_preds.loc[unchecked_edge['pairwise_id'], 'label'] == 1:
                                    true_positives_recovered += 1
                                labeling_budget -= 1
                            subgraphs = self.update_subgraphs(self.pairwise_matches_preds)
                    continue

                transitive_edges = full_data_utils.get_transitive_matches_subgraph(new_subgraph)
                transitive_edges = pd.DataFrame(transitive_edges, columns=['lid', 'rid', 'match_type'])
                transitive_edges = self.sort_lids_rids(transitive_edges)
                transitive_edges = transitive_edges.drop_duplicates(subset=['lid', 'rid'])
                transitive_edges['label'] = 0 # Add a dummy label to the transitive edges
                transitive_preds = transitive_edges.merge(all_transitive_preds, on=['lid', 'rid'], how='left')
                transitive_preds = transitive_preds.dropna(subset=['prob']) # Remove the transitive edges that have not been evaluated
                if len(transitive_edges) == 0:
                    # If we do not have any transitive pred, the edge has the potential to form a subgraph bigger than the subgraph_size_threshold (if added back to the matches_graph with other deleted edges),
                    # due to memory constraints we have not evaluated all the possible transitive edges of such subgraph, so we manually check the edge.
                    if self.args.manual_check:
                        if labeling_budget > 0:
                            # We manually check the edge
                            self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'manual_check_label'] = self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label']
                            self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'prob'] = self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label']
                            labeling_budget -= 1
                            if self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label'] == 1:
                                true_positives_recovered += 1
                                subgraphs = self.update_subgraphs(self.pairwise_matches_preds)
                    continue

                if transitive_preds['prob'].min() >= self.threshold:
                    # All the transitive edges are predicted as matches, we add the edge back to the matches_graph
                    self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'prob'] = 1
                    if self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label'] == 0:
                        false_positives_readded += 1
                    else:
                        true_positives_recovered += 1
                    # Update the subgraphs, remove both former subgraphs and add the new subgraph
                    if len(subgraph_with_lid) > 0:
                        if subgraph_with_lid[0] in subgraphs:
                            subgraphs.remove(subgraph_with_lid[0])
                    if len(subgraph_with_rid) > 0:
                        if subgraph_with_rid[0] in subgraphs:
                            subgraphs.remove(subgraph_with_rid[0])
                    subgraphs.append(new_subgraph)
                else:
                    # We manually check the edge
                    if self.args.manual_check:
                        if labeling_budget > 0:
                            self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'manual_check_label'] = self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label']
                            self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'prob'] = self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label']
                            labeling_budget -= 1
                            if self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label'] == 1:
                                true_positives_recovered += 1
                                # Update the subgraphs, remove both former subgraphs and add the new subgraph
                                if len(subgraph_with_lid) > 0:
                                    if subgraph_with_lid[0] in subgraphs:
                                        subgraphs.remove(subgraph_with_lid[0])
                                if len(subgraph_with_rid) > 0:
                                    if subgraph_with_rid[0] in subgraphs:
                                        subgraphs.remove(subgraph_with_rid[0])
                                subgraphs.append(new_subgraph)
                        else:
                            # We have run out of labeling budget, so we skip this edge
                            continue
        
        # Save the edge recovery results
        self.save_edge_recovery_results(false_positives_readded, true_positives_recovered, labeling_budget,
                                        pre_edge_recovery_true_positives, pre_edge_recovery_false_positives)

    def save_edge_recovery_results(self, false_positives_readded, true_positives_recovered, labeling_budget, 
                                    pre_edge_recovery_true_positives, pre_edge_recovery_false_positives):
        """
        Save the edge recovery results to a CSV file.
        """

        edge_recovery_results_path = os.path.join(self.post_finetuning_results_path, 'post_edge_recovery')
        if not os.path.exists(edge_recovery_results_path):
            os.makedirs(edge_recovery_results_path)

        # Save the post edge recovery results

        post_edge_recovery_true_positives = self.pairwise_matches_preds[(self.pairwise_matches_preds['prob'] >= self.threshold) & (self.pairwise_matches_preds['label'] == 1)].shape[0]
        post_edge_recovery_false_positives = self.pairwise_matches_preds[(self.pairwise_matches_preds['prob'] >= self.threshold) & (self.pairwise_matches_preds['label'] == 0)].shape[0]
        post_edge_recovery_true_negatives = self.pairwise_matches_preds[(self.pairwise_matches_preds['prob'] < self.threshold) & (self.pairwise_matches_preds['label'] == 0)].shape[0]
        post_edge_recovery_false_negatives = self.pairwise_matches_preds[(self.pairwise_matches_preds['prob'] < self.threshold) & (self.pairwise_matches_preds['label'] == 1)].shape[0]
        percent_checked = self.pairwise_matches_preds[self.pairwise_matches_preds['manual_check_label'] != -1].shape[0] / self.pairwise_matches_preds.shape[0]

        # Evaluate the final set of transitive matches
        matches_graph = full_data_utils.generate_matches_graph(self.pairwise_matches_preds, threshold=self.threshold)
        transitive_matches_preds = self.evaluate_transitive_edges(matches_graph)

        post_edge_recovery_results = {
            'false_positives_readded': false_positives_readded,
            'true_positives_recovered': true_positives_recovered,
            'labeling_budget_left': labeling_budget,
            'pre_edge_recovery_true_positives': pre_edge_recovery_true_positives,
            'pre_edge_recovery_false_positives': pre_edge_recovery_false_positives,
            'post_edge_recovery_true_positives': post_edge_recovery_true_positives,
            'post_edge_recovery_false_positives': post_edge_recovery_false_positives,
            'post_edge_recovery_true_negatives': post_edge_recovery_true_negatives,
            'post_edge_recovery_false_negatives': post_edge_recovery_false_negatives,
            'post_edge_recovery_positive_transitive_preds': transitive_matches_preds[transitive_matches_preds['prob'] >= self.threshold].shape[0],
            'post_edge_recovery_negative_transitive_preds': transitive_matches_preds[transitive_matches_preds['prob'] < self.threshold].shape[0],
            'percent_checked': percent_checked
        }
        post_edge_recovery_results_df = pd.DataFrame([post_edge_recovery_results])
        post_edge_recovery_results_df.to_csv(os.path.join(edge_recovery_results_path, 'post_edge_recovery.csv'), index=False)

        # Save also the post edge recovery pairwise_matches_preds and the post edge recovery matches.
        self.pairwise_matches_preds.to_csv(os.path.join(edge_recovery_results_path, 'pairwise_matches_preds.csv'), index=False)

        # Generate again the matches_graph with the transitive matches
        matches_graph = full_data_utils.generate_matches_graph(self.pairwise_matches_preds, threshold=self.threshold)

        matches_graph, _ = full_data_utils.generate_transitive_matches_graph(matches_graph,
                                                                                add_transitive_edges=True,
                                                                                results_path= self.finetune_results_path,
                                                                                subgraph_size_threshold=self.subgraph_size_threshold)
        # Save the matches graph 

        matches_graph_df = pd.DataFrame(matches_graph.edges(data=True), columns=['lid', 'rid', 'match_type'])
        matches_graph_df.to_csv(os.path.join(edge_recovery_results_path, 'post_graph_cleanup_matches.csv'), index=False)



    def get_new_subgraph(self, subgraph_with_lid, subgraph_with_rid, lid_rid_subgraph):
        """
        Create a new subgraph that contains all nodes and edges of the previous 2 subgraphs along with the (lid, rid) edge.
        """
        new_subgraph = nx.Graph()
        # Either subgraph_with_lid or subgraph_with_rid may be empty (when the record is not linked to any other record)
        if len(subgraph_with_lid) > 0:
            new_subgraph.add_nodes_from(subgraph_with_lid[0].nodes(data=True))
            new_subgraph.add_edges_from(subgraph_with_lid[0].edges(data=True))
        if len(subgraph_with_rid) > 0:
            new_subgraph.add_nodes_from(subgraph_with_rid[0].nodes(data=True))
            new_subgraph.add_edges_from(subgraph_with_rid[0].edges(data=True))
        
        new_subgraph.add_nodes_from(lid_rid_subgraph.nodes(data=True))
        new_subgraph.add_edges_from(lid_rid_subgraph.edges(data=True))

        return new_subgraph


    def get_all_transitive_preds(self, deleted_edges):
        """
        Predict all the transitive edges of the matches_graph created by adding back all the deleted edges. 
        We will use these predictions to decide which edges to add back to the matches_graph in the edge recovery process.
        """
        pairwise_preds = copy.deepcopy(self.pairwise_matches_preds)
        # Add the deleted edges back to the pairwise_preds
        pairwise_preds.loc[deleted_edges.index, 'prob'] = 1
        matches_graph = full_data_utils.generate_matches_graph(pairwise_preds, threshold=self.threshold)
        transitive_matches = full_data_utils.get_transitive_matches_edge_recovery(matches_graph, subgraph_size_threshold=self.subgraph_size_threshold)
        transitive_matches_df = pd.DataFrame(transitive_matches, columns=['lid', 'rid', 'match_type'])
        transitive_matches_df = self.sort_lids_rids(transitive_matches_df)
        transitive_matches_df = transitive_matches_df.drop_duplicates(subset=['lid', 'rid'])
        transitive_matches_df['label'] = 0 # Add a dummy label to the transitive edges

        # Evaluate the transitive matches

        transitive_matches_preds = self.evaluate_edges(transitive_matches_df)

        return transitive_matches_preds

    def LLM_labeling_edge_recovery(self, lid, rid):
        """
        We label the edge with the LLM
        """
        # Load the LLM model
        self.load_LLM_model()

        lid_record = self.model.dataset.tokenized_data.loc[lid]['concat_data']
        rid_record = self.model.dataset.tokenized_data.loc[rid]['concat_data']

        # We use the LLM to label the edge
        LLM_label = label_record_pair_with_LLM(self.LLM_model, self.LLM_tokenizer, self.LLM_model_max_length, lid_record, rid_record)

        # Delete the LLM model to free up memory
        del self.LLM_model
        
        return LLM_label
    
    def process_LLM_labeling(self, deleted_edge, LLM_label, true_positives_recovered, false_positives_readded):
        self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'manual_check_label'] = LLM_label
        self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'prob'] = LLM_label
        if LLM_label == 1 and self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label'] == 1:
            true_positives_recovered += 1
        elif LLM_label == 1 and self.pairwise_matches_preds.loc[deleted_edge['pairwise_id'], 'label'] == 0:
            false_positives_readded += 1

        return true_positives_recovered, false_positives_readded



    ###########################################################################

    # Blocking Utils

    ###########################################################################

    def get_tknzd_records_and_overlap_indicators(self, test_entity_data):
        tokenized_records = self.model.dataset.get_tokenized_data()
        # The tokenized records are indexed by the id of the raw data
        tokenized_test_records = tokenized_records[tokenized_records.index.isin(test_entity_data['id'])]
        # Generate the list of all tokens seen in the test records
        tmp_list = tokenized_test_records['tokenized'].apply(lambda x: list(set(x)))
        tmp_list = tmp_list.apply(lambda x: list(set(x)))
        all_tokens = np.array(list(set(string for sublist in tmp_list for string in sublist)))
        # Index structure for much faster lookup of the index positions of every token in the all_tokens list
        # (so that we do not have to call all_tokens.index but rather get a O(1) lookup)
        index_lookup = {value: i for i, value in enumerate(all_tokens)}
        # Generating a sparse matrix with (n_records, n_tokens), where a 1 at (recordX, tokenY) indicates,
        # that recordX contains the tokenY
        #
        data = []
        row = []
        col = []
        for i, (record_id, tokenized_record) in tqdm(enumerate(tokenized_test_records.iterrows()),
                                                        total=tokenized_test_records.shape[0],
                                                        desc='Building indices matrix'):
            token_indexes_in_record = sorted(set([index_lookup[t] for t in tokenized_record['tokenized']]))

            n_tokens = len(token_indexes_in_record)
            data.extend([True for _ in range(n_tokens)])
            row.extend([i for _ in range(n_tokens)])
            col.extend(token_indexes_in_record)
        indicators = csr_matrix((data, (row, col)), shape=(tokenized_test_records.shape[0], len(all_tokens)),
                                dtype=np.int8)
        return indicators, tokenized_test_records

    def get_top_overlap_idx(self, i, indicators, test_entity_data):
            lookup = np.array(indicators[i, :].dot(indicators.transpose()).todense())[0]
            # Set all records from the same data source to zero, because we only want matches with other data sources
            current_data_source = test_entity_data.iloc[i]['data_source_id']
            multiplication_mask = np.ones(test_entity_data.shape[0], dtype=np.int8) - \
                np.array(test_entity_data['data_source_id'] == current_data_source)
            lookup *= multiplication_mask
            top_overlap_idx = np.argpartition(lookup, -self.number_of_candidates)[-self.number_of_candidates:]
            return top_overlap_idx

    def get_top_overlap_idx_one_source(self, i, indicators, test_entity_data):
            lookup = np.array(indicators[i, :].dot(indicators.transpose()).todense())[0]
            top_overlap_idx = np.argpartition(lookup, -self.number_of_candidates)[-self.number_of_candidates:]
            return top_overlap_idx
    


            
