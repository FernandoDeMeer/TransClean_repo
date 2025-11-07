import os
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import copy 
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.getcwd())

from src.helpers.path_helper import *
from src.CLER.utils import *
from src.CLER.model import *
from src.CLER.dataset import *
from src.data import full_data_utils

def calculate_metrics(labels, preds):
    true_positives = np.sum((labels == 1) & (preds == 1))
    false_positives = np.sum((labels == 0) & (preds == 1))
    false_negatives = np.sum((labels == 1) & (preds == 0))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return true_positives, false_positives, false_negatives, precision, recall, f1

def load_CLER_pairwise_preds(dataset_name, CLER_experiment_name):
    cler_preds_path = dataset_results_folder_path__with_subfolders([dataset_name, CLER_experiment_name])
    cler_preds = pd.read_csv(os.path.join(cler_preds_path, 'pairwise_matches_preds.csv'))
    cler_preds = sort_lids_rids(cler_preds)
    cler_preds = cler_preds.drop_duplicates(subset=['lid', 'rid'])
    # CLER predictions include both (lid, rid) and (rid, lid), this can lead to issues when calculating metrics 
    # (such as which prediction to choose if (lid, rid) and (rid, lid) have different probabilities?). To avoid this issue we save the sorted and deduplicated predictions
    cler_preds['match_type'] = 'CLER_match'
    cler_preds.to_csv(os.path.join(cler_preds_path, 'pairwise_matches_preds.csv'), index=False)
    return cler_preds, cler_preds_path

def sort_lids_rids(input_df):
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
    
def eval_translate_ids(df, idxs_df):
    # Create a dictionary to map the original ids to the positional indices
    id2idx = {}
    for idx, id in enumerate(df.index):
        id2idx[id] = idx
    # Translate the ids
    idxs_df['ltable_id'] = idxs_df['ltable_id'].apply(lambda x: id2idx[x])
    idxs_df['rtable_id'] = idxs_df['rtable_id'].apply(lambda x: id2idx[x])

    return idxs_df, id2idx

def eval_translate_ids_back(df, id2idx):
    # Translate the positional indices back to the original ids

    # Reverse the dictionary
    idx2id = {v: k for k, v in id2idx.items()}
    # Translate the ids back (from positional indices in the index to the original ids)
    for i, row in df.iterrows():
        df.at[i, 'ltable_id'] = idx2id[row['ltable_id']]
        df.at[i, 'rtable_id'] = idx2id[row['rtable_id']]

    return df

def get_CLER_metrics(pos_cler_preds, transitive_matches, post_graph_cleanup_matches, ground_truth):
    # Merge the ground truth with the pairwise matches/transitive matches/post graph cleanup matches, to do this we set the (lid, rid) as the index for each df
    ground_truth = ground_truth.set_index(['lid', 'rid'])
    pos_cler_preds = pos_cler_preds.set_index(['lid', 'rid'])
    post_graph_cleanup_matches = post_graph_cleanup_matches.set_index(['lid', 'rid'])
    transitive_matches = transitive_matches.set_index(['lid', 'rid'])

    # Do the merges
    pos_cler_preds = pos_cler_preds.join(ground_truth, how='outer')
    post_graph_cleanup_matches = post_graph_cleanup_matches.join(ground_truth, how='outer')
    transitive_matches = transitive_matches.join(ground_truth, how='outer')
    # Set the NaNs to 0
    pos_cler_preds['label'] = pos_cler_preds['label'].fillna(0)
    pos_cler_preds['prob'] = pos_cler_preds['prob'].fillna(0)
    post_graph_cleanup_matches['label'] = post_graph_cleanup_matches['label'].fillna(0)
    post_graph_cleanup_matches['prob'] = post_graph_cleanup_matches['prob'].fillna(0)
    transitive_matches['label'] = transitive_matches['label'].fillna(0)
    transitive_matches['prob'] = transitive_matches['prob'].fillna(0)

    # Calculate precision, recall, f1 for each of the match types
    true_pos_pairwise, false_pos_pairwise, false_neg_pairwise , pos_cler_preds_precision, pos_cler_preds_recall, pos_cler_preds_f1 = calculate_metrics(pos_cler_preds['label'], pos_cler_preds['prob'])
    print(f'True positives, false positives, false negatives for the pairwise matches: {true_pos_pairwise}, {false_pos_pairwise}, {false_neg_pairwise}')
    print(f'Precision, recall, f1 for the positive CLER predictions: {pos_cler_preds_precision}, {pos_cler_preds_recall}, {pos_cler_preds_f1}')
    _,_,_, post_graph_cleanup_matches_precision, post_graph_cleanup_matches_recall, post_graph_cleanup_matches_f1 = calculate_metrics(post_graph_cleanup_matches['label'], post_graph_cleanup_matches['prob'])
    print(f'Precision, recall, f1 for the post graph cleanup matches: {post_graph_cleanup_matches_precision}, {post_graph_cleanup_matches_recall}, {post_graph_cleanup_matches_f1}')
    _,_,_, transitive_matches_precision, transitive_matches_recall_, transitive_matches_f1 = calculate_metrics(transitive_matches['label'], transitive_matches['prob'])
    print(f'Precision, recall, f1 for the transitive matches: {transitive_matches_precision}, {transitive_matches_recall_}, {transitive_matches_f1}')

    metrics_dict = {'true_pos_pairwise': true_pos_pairwise, 'false_pos_pairwise': false_pos_pairwise, 'false_neg_pairwise': false_neg_pairwise, 
                    'pos_cler_preds_precision': pos_cler_preds_precision, 'pos_cler_preds_recall': pos_cler_preds_recall, 
                    'pos_cler_preds_f1': pos_cler_preds_f1, 'post_graph_cleanup_matches_precision': post_graph_cleanup_matches_precision, 
                    'post_graph_cleanup_matches_recall': post_graph_cleanup_matches_recall, 'post_graph_cleanup_matches_f1': post_graph_cleanup_matches_f1, 
                    'transitive_matches_precision': transitive_matches_precision, 'transitive_matches_recall': transitive_matches_recall_,
                    'transitive_matches_f1': transitive_matches_f1}
    results_path = dataset_results_folder_path__with_subfolders([args.dataset_name, args.CLER_experiment_name])
    # Save the metrics to a csv
    metrics_df = pd.DataFrame(metrics_dict, index=[0])
    metrics_df.to_csv(os.path.join(results_path, 'CLER_metrics.csv'), index=False)

def get_transitive_matches(input_graph, subgraph_size_threshold=50):
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


def predict_transitive_matches(model, matches_graph, dataset_path):
    transitive_matches = get_transitive_matches(matches_graph)
    transitive_matches = pd.DataFrame(transitive_matches, columns=['lid', 'rid', 'match_type'])
    transitive_matches = sort_lids_rids(transitive_matches)
    transitive_matches.drop_duplicates(subset=['lid', 'rid'], inplace=True)
    transitive_matches.drop(columns=['match_type'], inplace=True)
    transitive_matches = transitive_matches.rename(columns = {'lid': 'ltable_id', 'rid': 'rtable_id'})
    transitive_matches['label'] = 0 #Dummy label for transitive matches
    transitive_ids = np.concatenate([transitive_matches['ltable_id'].values, transitive_matches['rtable_id'].values])


    # Load the dataset data
    dataset_raw_df = pd.read_csv(dataset_path) 
    if 'companies' in dataset_path:
        dataset_raw_df = dataset_raw_df.drop(columns=['data_source_id', 'inserted', 'last_modified', 'external_id'])
    dataset_raw_df['id'] = dataset_raw_df.index 
    transitive_records_df = dataset_raw_df[dataset_raw_df['id'].isin(transitive_ids)]
    transitive_records_df = transitive_records_df.drop(columns=['id'])

    transitive_matches, transitive_matches_ids_dict = eval_translate_ids(transitive_records_df, transitive_matches)

    # Get the entity list and attribute list
    entity_list = transitive_records_df.values
    attr_list = list(transitive_records_df.columns)
    trans_set = GTDatasetWithLabel(transitive_matches.values, entity_list, entity_list, attr_list, lm='roberta', concat=True, shuffle=False)

    trans_preds = pred_w_CLER(model, trans_set, batch_size=64)

    transitive_matches['pred'] = trans_preds[0]
    transitive_matches = eval_translate_ids_back(transitive_matches, transitive_matches_ids_dict)

    return transitive_matches

def visualize_CLER_preds(pos_cler_preds, transitive_matches, ground_truth, results_path, args):
    # Merge the ground truth with the pos_cler_preds in order to get the True and False Positives of the CLER model
    ground_truth = ground_truth.set_index(['lid', 'rid'])
    pos_cler_preds = pos_cler_preds.set_index(['lid', 'rid'])
    pos_cler_preds = pos_cler_preds.join(ground_truth, how='left')
    pos_cler_preds['label'] = pos_cler_preds['label'].fillna(0)

    # Get the numbers of True Positives and False Positives of the CLER model
    true_positives = pos_cler_preds[(pos_cler_preds['label'] == 1) & (pos_cler_preds['prob'] == 1)].shape[0]
    false_positives = pos_cler_preds[(pos_cler_preds['label'] == 0) & (pos_cler_preds['prob'] == 1)].shape[0]

    # Get the numbers of positive and negative predictions of the transitive matches
    pos_transitive_matches = transitive_matches[transitive_matches['pred'] == 1].shape[0]
    neg_transitive_matches = transitive_matches[transitive_matches['pred'] == 0].shape[0]

    # Put all the data in a DataFrame
    df = pd.DataFrame({'True Positives': [true_positives], 'False Positives': [false_positives], 
                       'Positive Transitive Matches': [pos_transitive_matches], 'Negative Transitive Matches': [neg_transitive_matches]})
    # Get the labeling budget for the x-axis label from the results path
    CLER_labeling_budget = results_path.split('CLER_')[1].split('_')[0]
    x_label = ['Labeling Budget {}'.format(CLER_labeling_budget)]
    df['idx'] = x_label

    fig, ax1 = plt.subplots(figsize=(8, 10))
    tidy = df.melt(id_vars='idx').rename(columns=str.title)
    sns.barplot(x = 'Idx', y = 'Value', hue = 'Variable', data = tidy, ax=ax1)
    sns.despine(fig)

    
    ax1.set_title('CLER Pairwise Predictions and Transitive Matches')

    # Make the y scale logarithmic
    plt.yscale('log')
    # Add 'log-scale' to the y-axis label
    ax1.set_ylabel('Number of Matches (log-scale)')
    # Put the legend on the upper right side of the Fig
    plt.legend(loc='upper right',  ncol = 1, fancybox=True, shadow=True)
    # Add a grid to the plot
    plt.grid(axis='y')
    # Save the plot
    plt.savefig(os.path.join(results_path, 'CLER_preds_visualization.png'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CLER predictions')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--CLER_experiment_name', type=str, help='Name of the CLER experiment', required=True)
    parser.add_argument('--ground_truth_path', type=str, help='Path to the ground truth file', required=True)

    args = parser.parse_args()
    dataset_path_dict = {'synthetic_companies': 'data/raw/synthetic_data/seed_0/synthetic_companies_dataset_seed_0_size_868254_sorted.csv',
                         'wdc': 'data/raw/wdc80_pair/wdc_80pair.csv',
                         'camera': 'data/raw/camera/camera.csv',
                         'monitor': 'data/raw/monitor/monitor.csv',
                         }

    cler_pairwise_preds, cler_preds_path = load_CLER_pairwise_preds(args.dataset_name, args.CLER_experiment_name)

    # Make the matches graph with the CLER predictions and adding the transitive matches
    matches_graph = full_data_utils.generate_matches_graph(cler_pairwise_preds, threshold=0.999) #The threshold here doesn't matter since CLER returns 0/1 predictions
    matches_graph, transitive_matches = full_data_utils.generate_transitive_matches_graph(matches_graph, add_transitive_edges=False, results_path=cler_preds_path, subgraph_size_threshold=50)

    # Save the transitive matches
    transitive_matches = pd.DataFrame(transitive_matches, columns=['lid', 'rid', 'match_type'])
    transitive_matches['prob'] = 1
    transitive_matches = sort_lids_rids(transitive_matches)
    transitive_matches = transitive_matches.drop_duplicates(subset=['lid', 'rid'])
    transitive_matches.to_csv(os.path.join(cler_preds_path, 'pre_cleanup_transitive_matches.csv'), index=False)

    # Add the transitive matches to the original CLER predictions and save them as the post graph cleanup matches (CLER does not carry out any graph cleanup)
    pos_cler_preds = cler_pairwise_preds[cler_pairwise_preds['prob'] == 1]
    post_graph_cleanup_matches = pd.concat([pos_cler_preds, transitive_matches])
    post_graph_cleanup_matches.reset_index(drop=True, inplace=True)
    post_graph_cleanup_matches.to_csv(os.path.join(cler_preds_path, 'post_graph_cleanup_matches.csv'), index=False)

    # Load the ground truth

    ground_truth = pd.read_csv(args.ground_truth_path)
    ground_truth = ground_truth[ground_truth['label'] == 1]

    get_CLER_metrics(pos_cler_preds, transitive_matches, post_graph_cleanup_matches, ground_truth)


    # Load the CLER model and predict the transitive matches
    # First check if the transitive matches have already been predicted
    CLER_results_path = dataset_results_folder_path__with_subfolders([args.dataset_name, args.CLER_experiment_name])
    if not os.path.exists(os.path.join(CLER_results_path, 'transitive_matches_preds.csv')):
        model = load_CLER_matcher(args.CLER_experiment_name)
        transitive_matches = predict_transitive_matches(model, matches_graph, dataset_path_dict[args.dataset_name])
        transitive_matches.to_csv(os.path.join(CLER_results_path, 'transitive_matches_preds.csv'), index=False)
    else:
        transitive_matches = pd.read_csv(os.path.join(CLER_results_path, 'transitive_matches_preds.csv'))

    visualize_CLER_preds(pos_cler_preds, transitive_matches, ground_truth, CLER_results_path, args)