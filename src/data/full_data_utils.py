import os
import pickle
import pandas as pd
import scipy
from tqdm.auto import tqdm
import numpy as np
import networkx as nx
import copy

from src.data.dataset import ExperimentDataset
from src.helpers.path_helper import *


ID_ATTRIBUTES = {'isin': 'ISIN', 'cusip': 'CUSIP', 'valor': 'VALOR', 'sedol': 'SEDOL'}


def load_syn_master_mapping(dataset_name):

    if 'synthetic_companies' in dataset_name:
        master_df = pd.read_csv('data/raw/synthetic_data/seed_0/companies_master_mapping_seed_0.csv')

    return master_df

def get_train_val_test_split(mapping_df, **kwargs):

    number_of_gen_ids = kwargs.get('number_of_gen_ids', None)
    if number_of_gen_ids:
        gen_ids = mapping_df['gen_id'].unique()[:number_of_gen_ids]
    else:
        gen_ids = mapping_df['gen_id'].unique()

    train, val, test = np.split(gen_ids, [int(.6 * len(gen_ids)), int(.8 * len(gen_ids))])

    split_dict = {'train': train,
                  'val': val,
                  'test': test}
    return split_dict


def add_id_to_records(records_df, raw_df,) -> pd.DataFrame:
    # Assign to each (test) record its corresponding id value from the raw data

    records_df = records_df.astype({'external_id': 'str', 'data_source_id': 'str'})
    raw_df = raw_df.astype({'external_id': 'str', 'data_source_id': 'str'})
    return records_df.merge(raw_df[['id', 'external_id', 'data_source_id']], on=['external_id', 'data_source_id'], how='left')

def generate_matches_graph(pairwise_matches_preds: pd.DataFrame, threshold: float = 0.999) -> nx.Graph:

    positive_matches_df = pairwise_matches_preds[pairwise_matches_preds['prob'] > threshold]

    try:
        matches_graph = nx.from_pandas_edgelist(positive_matches_df, 'lid', 'rid', ['prob', 'match_type'])
    except:
        matches_graph = nx.from_pandas_edgelist(positive_matches_df, 'lid', 'rid', ['prob'])
    return matches_graph

def generate_transitive_matches_graph(input_graph, add_transitive_edges = True, results_path = None, subgraph_size_threshold = 50):
    """
    Generates the transitive matches of the input graph. In case the subgraph is too big, we dont add its transitive edges to the matches_graph due to memory constraints and save its size to then calculate
    later scores in get_scores_matching.py.
    """

    matches_graph = copy.deepcopy(input_graph)

    subgraphs = list(nx.connected_components(matches_graph))

    transitive_matches = []

    big_subgraph_sizes = [len(c) for c in subgraphs if len(c) > subgraph_size_threshold]

    if len(big_subgraph_sizes) > 0 and results_path is not None:
        # We save the size of big subgraphs
        big_subgraph_sizes_df = pd.DataFrame(big_subgraph_sizes, columns=['size'])
        big_subgraph_sizes_df.to_csv(os.path.join(results_path, 'big_subgraph_sizes.csv'), index=False)

    for subgraph_idx, c in enumerate(subgraphs):
        subgraph_nodes = list(c)

        if len(subgraph_nodes) > subgraph_size_threshold:
            # If the subgraph is too big, we dont add its transitive edges to the matches_graph due to memory constraints, instead we will count
            # the transitive edges later on get_scores_matching.py
            continue

        # Add edges between all nodes of the subgraph to the matches_graph
        for lid in subgraph_nodes:
            for rid in subgraph_nodes:
                if lid != rid:
                    if not matches_graph.has_edge(lid, rid) and not matches_graph.has_edge(rid, lid) and add_transitive_edges:
                        matches_graph.add_edge(lid, rid, prob=0, match_type='transitive_match')
                    elif not matches_graph.has_edge(lid, rid) and not matches_graph.has_edge(rid, lid):
                        transitive_matches.append((lid, rid, 'transitive_match'))

    return matches_graph, transitive_matches

def add_transitive_edges_to_graph(input_graph):
    """
    Adds the transitive edges to the input graph.
    """

    matches_graph = copy.deepcopy(input_graph)

    subgraphs = list(nx.connected_components(matches_graph))

    for subgraph_idx, c in enumerate(subgraphs):
        subgraph_nodes = list(c)

        # Add the transitive edges between all nodes of the subgraph to the matches_graph
        for lid in subgraph_nodes:
            for rid in subgraph_nodes:
                if lid != rid:
                    if not matches_graph.has_edge(lid, rid) and not matches_graph.has_edge(rid, lid):
                        matches_graph.add_edge(lid, rid, prob=0, match_type='transitive_match')

    return matches_graph



def get_transitive_matches_finetuning(input_graph, subgraph_size_threshold = 50):
    """
    Generates the list of transitive matches of the input graph for the fine-tuning process. In case a subgraph is too big,
    we output only a (random) subset of its transitive matches.
    """

    matches_graph = copy.deepcopy(input_graph)

    subgraphs = list(nx.connected_components(matches_graph))

    transitive_matches = []

    for subgraph_idx, c in enumerate(subgraphs):
        subgraph_nodes = list(c)

        if len(subgraph_nodes) > subgraph_size_threshold:
            # If the subgraph is too big, we collect a random subset of its transitive edges
            transitive_matches_subset = []
            random_nodes = np.random.choice(subgraph_nodes, subgraph_size_threshold, replace=False)
            for lid in random_nodes:
                for rid in random_nodes:
                    if lid != rid:
                        if not matches_graph.has_edge(lid, rid) and not matches_graph.has_edge(rid, lid) and (lid, rid, 'transitive_match') not in transitive_matches_subset:
                            transitive_matches_subset.append((lid, rid, 'transitive_match'))
            transitive_matches.extend(transitive_matches_subset)
        else:
            # Add edges between all nodes of the subgraph to the matches_graph
            for lid in subgraph_nodes:
                for rid in subgraph_nodes:
                    if lid != rid:
                        if not matches_graph.has_edge(lid, rid) and not matches_graph.has_edge(rid, lid):
                            transitive_matches.append((lid, rid, 'transitive_match'))

    return transitive_matches

def get_all_transitive_matches(input_graph):
    """
    Generates the full list of transitive matches of the input graph.
    """

    matches_graph = copy.deepcopy(input_graph)

    subgraphs = list(nx.connected_components(matches_graph))

    transitive_matches = []

    for subgraph_idx, c in enumerate(subgraphs):
        subgraph_nodes = list(c)

        # Add the transitive edges between all nodes of the subgraph to the matches_graph
        for lid in subgraph_nodes:
            for rid in subgraph_nodes:
                if lid != rid:
                    if not matches_graph.has_edge(lid, rid) and not matches_graph.has_edge(rid, lid):
                        transitive_matches.append((lid, rid, 'transitive_match'))

    return transitive_matches

def get_transitive_matches_subgraph(subgraph):
    """
    Generates the transitive matches of a subgraph.
    """
    transitive_matches = []
    for lid in subgraph:
        for rid in subgraph:
            if lid != rid:
                if not subgraph.has_edge(lid, rid) and not subgraph.has_edge(rid, lid) and (lid, rid) not in transitive_matches and (rid, lid) not in transitive_matches:
                    transitive_matches.append((lid, rid, 'transitive_match'))
                    
    return transitive_matches

def get_transitive_matches_edge_recovery(input_graph, subgraph_size_threshold = 50):
    """
    Generates the transitive matches of the input graph for the edge recovery process. In case a subgraph is bigger than the threshold, we dont consider its transitive edges.
    """
    
    matches_graph = copy.deepcopy(input_graph)

    subgraphs = list(nx.connected_components(matches_graph))

    transitive_matches = []

    for subgraph_idx, c in enumerate(subgraphs):
        subgraph_nodes = list(c)

        if len(subgraph_nodes) >= subgraph_size_threshold:
            # If the subgraph is too big, we dont consider its transitive edges due to memory constraints, 
            # we will manually check the deleted edges corresponding to this subgraph if the labeling budget allows it.
            continue

        # Add edges between all nodes of the subgraph to the matches_graph
        for lid in subgraph_nodes:
            for rid in subgraph_nodes:
                if lid != rid:
                    if not matches_graph.has_edge(lid, rid) and not matches_graph.has_edge(rid, lid):
                        transitive_matches.append((lid, rid, 'transitive_match'))

    return transitive_matches
