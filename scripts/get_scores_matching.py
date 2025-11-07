import os
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pickle

from scipy.sparse import coo_matrix
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.helpers.path_helper import *




def get_scores_args():
    parser = argparse.ArgumentParser(description='Calculate the scores of a records matching experiment')

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--experiment_names_list', action='append', required=True)
    parser.add_argument('--ground_truth_path', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.999, required=False)
    parser.add_argument('--non_positional_ids', action='store_true', required=False)
    parser.add_argument('--post_edge_recovery', action='store_true', required=False)
    parser.add_argument('--labeling_budgets_list', action='append', required=False) # Used to produce a graph of labeling budget vs f1 score
    parser.add_argument('--add_CLER_scores', action='store_true', required=False)
    parser.add_argument('--CLER_post_cleanup_list', action='append', required=False) # Used to add the post-finetuning cleanup of different CLER models the labeling budget vs f1 score graph
    args = parser.parse_args()

    return args


def get_scores(matching_folder_path, ground_truth_file_path, dataset_name, threshold, non_positional_ids = False, post_edge_recovery = False, labeling_budget = None):
    """
    Computes all the scores of a given matching, including pairwise scores and pre/post graph cleanup entity group matching scores.
    """
    scores_dict = {}

    # Load the ground truth

    ground_truth = pd.read_csv(ground_truth_file_path)
    # Filter out the negative pairs of the ground truth
    ground_truth = filter_ground_truth_pairs_df(ground_truth)
    # ground_truth = add_transitive_pairs(ground_truth)

    # Load the pairwise predictions, the subgraph size list (if applicable), the pre-graph cleanup transitive matches and all the post-graph cleanup matches

    pairwise_matches_preds = pd.read_csv(os.path.join(matching_folder_path, 'pairwise_matches_preds.csv'), header=0)

    # Calculate the recall of the candidate pairs (i.e. the number of pairs in the ground truth that are also in the pairwise predictions)
    true_positive_candidate_pairs = ground_truth[ground_truth['lid'].isin(pairwise_matches_preds['lid']) & ground_truth['rid'].isin(pairwise_matches_preds['rid'])]
    true_positive_candidate_pairs = true_positive_candidate_pairs[true_positive_candidate_pairs['label'] == 1]
    scores_dict['recall_candidate_pairs'] = len(true_positive_candidate_pairs) / len(ground_truth[ground_truth['label'] == 1])

    # Check for the big subgraphs size list
    if os.path.isfile(os.path.join(matching_folder_path, 'big_subgraph_sizes.csv')):
        big_subgraph_sizes_df= pd.read_csv(os.path.join(matching_folder_path, 'big_subgraph_sizes.csv'), header=0)
        big_subgraph_sizes = big_subgraph_sizes_df['size'].values.tolist()
    else:
        big_subgraph_sizes = []

    pre_cleanup_transitive_matches = pd.read_csv(os.path.join(matching_folder_path, 'pre_cleanup_transitive_matches.csv'), header=0)

    if post_edge_recovery:
        if labeling_budget is not None:
            matching_folder_path = os.path.join(matching_folder_path, 'labeling_budget_{}'.format(labeling_budget))
        matching_folder_path = os.path.join(matching_folder_path, 'post_edge_recovery')
        post_graph_cleanup_matches = pd.read_csv(os.path.join(matching_folder_path, 'post_graph_cleanup_matches.csv'), header=0)
        post_graph_cleanup_preds = pd.read_csv(os.path.join(matching_folder_path, 'pairwise_matches_preds.csv'), header=0)
        # Get the post graph cleanup true positives and false positives
        scores_dict['post_cleanup_true_positives'] = post_graph_cleanup_preds[(post_graph_cleanup_preds['prob'] >= threshold) & (post_graph_cleanup_preds['label'] == 1)].shape[0]
        scores_dict['post_cleanup_false_positives'] = post_graph_cleanup_preds[(post_graph_cleanup_preds['prob'] >= threshold) & (post_graph_cleanup_preds['label'] == 0)].shape[0]

    else:
        post_graph_cleanup_matches = pd.read_csv(os.path.join(matching_folder_path, 'post_graph_cleanup_matches.csv'), header=0)
                                                                                                         

    # Preprocess the pairwise predictions

    pairwise_matches_preds = filter_pairs_df(pairwise_matches_preds, threshold = threshold)

    # If we have non-positional ids, we need to trasnform them to positional ids (otherwise the sparse matrices will be too big)

    if non_positional_ids:
        ground_truth = transform_ids_to_positional_ids(ground_truth, dataset_name)
        pairwise_matches_preds = transform_ids_to_positional_ids(pairwise_matches_preds, dataset_name)
        pre_cleanup_transitive_matches = transform_ids_to_positional_ids(pre_cleanup_transitive_matches, dataset_name)
        post_graph_cleanup_matches = transform_ids_to_positional_ids(post_graph_cleanup_matches, dataset_name)

    # Find the max lid/rid 

    max_id = get_max_id(ground_truth, pairwise_matches_preds, pre_cleanup_transitive_matches, post_graph_cleanup_matches)

    # Construct the sparse coo-matrices of pairs and labels for the ground truth and the predictions
 
    ground_truth_sparse = construct_sparse_matrix(ground_truth, max_id)
    pairwise_preds = construct_sparse_matrix(pairwise_matches_preds, max_id)
    pre_cleanup_transitive_matches_sparse = construct_sparse_matrix(pre_cleanup_transitive_matches, max_id)
    post_graph_cleanup_matches = construct_sparse_matrix(post_graph_cleanup_matches, max_id)

    # Build the sparse matrix for the pre cleanup entity group matching score

    pre_clean_up_matches = pairwise_preds + pre_cleanup_transitive_matches_sparse

    # For each matches matrix, get true positives, false positives and false negatives and calculate the pairwise scores

    for matches_matrix in [pairwise_preds, pre_clean_up_matches, post_graph_cleanup_matches]:

        true_positives, false_positives = get_true_and_false_positives(ground_truth_sparse=ground_truth_sparse,
                                                                       predictions_sparse=matches_matrix)

        false_negatives = get_false_negatives(ground_truth_sparse=ground_truth_sparse,
                                              predictions_sparse=matches_matrix)

        precision, recall, f1_score = get_pairwise_scores(true_positives=true_positives,
                                                        false_positives=false_positives,
                                                        false_negatives=false_negatives,
                                                        big_subgraph_sizes=big_subgraph_sizes,
                                                        pairwise_pre_cleanup_scores=(matches_matrix is pre_clean_up_matches and len(big_subgraph_sizes) > 0))
        
        if matches_matrix is pairwise_preds:
            scores_dict['pairwise_preds'] = {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'true_positives': len(true_positives), 'false_positives': len(false_positives), 'false_negatives': len(false_negatives)}
            scores_dict['removed_edges'] = {'pre_cleanup_true_positives': len(true_positives), 'pre_cleanup_false_positives': len(false_positives), 'true_positives_change': None, 'false_positives_change': None}
        elif matches_matrix is pre_clean_up_matches:
            scores_dict['pre_cleanup_matches'] = {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'true_positives': len(true_positives), 'false_positives': len(false_positives), 'false_negatives': len(false_negatives)}
        elif matches_matrix is post_graph_cleanup_matches:
            scores_dict['post_graph_cleanup_matches'] = {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'true_positives': len(true_positives), 'false_positives': len(false_positives), 'false_negatives': len(false_negatives)}

    if post_edge_recovery:
        # Calculate the number of removed edges in the pre and post graph cleanup matches
        if scores_dict['removed_edges']['pre_cleanup_true_positives'] != 0:
            scores_dict['removed_edges']['true_positives_change'] = 100*(scores_dict['removed_edges']['pre_cleanup_true_positives'] - scores_dict['post_cleanup_true_positives'])/scores_dict['removed_edges']['pre_cleanup_true_positives']
        else:
            scores_dict['removed_edges']['true_positives_change'] = 0
        if scores_dict['removed_edges']['pre_cleanup_false_positives'] != 0:
            scores_dict['removed_edges']['false_positives_change'] = 100*(scores_dict['removed_edges']['pre_cleanup_false_positives'] - scores_dict['post_cleanup_false_positives'])/scores_dict['removed_edges']['pre_cleanup_false_positives']
        else:
            scores_dict['removed_edges']['false_positives_change'] = 0
    # Calculate graph metrics for the pre and post graph cleanup matches

    scores_dict['subgraph_purity'] = {'pre-cleanup': None, 'post-cleanup': None}
    scores_dict['number_of_subgraphs'] = {'pre-cleanup': None, 'post-cleanup': None}

    for matches_matrix in [pre_clean_up_matches, post_graph_cleanup_matches]:

        # Generate the matches graph from all the positive predictions
        matches_graph = nx.from_scipy_sparse_matrix(matches_matrix, create_using=nx.Graph())
        subgraph_purity = get_subgraph_purity(matches_graph, ground_truth_sparse)
        number_of_subgraphs = nx.number_connected_components(matches_graph)

        if matches_matrix is pre_clean_up_matches:
            scores_dict['subgraph_purity']['pre-cleanup'] = subgraph_purity
            scores_dict['number_of_subgraphs']['pre-cleanup'] = number_of_subgraphs
        elif matches_matrix is post_graph_cleanup_matches:
            scores_dict['subgraph_purity']['post-cleanup'] = subgraph_purity
            scores_dict['number_of_subgraphs']['post-cleanup'] = number_of_subgraphs

    return scores_dict

def get_max_id(ground_truth, pairwise_matches_preds, pre_cleanup_transitive_matches, post_graph_cleanup_matches):
    # Get the maximum id from the ground truth and all sets of matches
    max_id = max(ground_truth['lid'].max(), ground_truth['rid'].max(), 
                 pairwise_matches_preds['lid'].max(), pairwise_matches_preds['rid'].max(), 
                 pre_cleanup_transitive_matches['lid'].max(), pre_cleanup_transitive_matches['rid'].max(), 
                 post_graph_cleanup_matches['lid'].max(), post_graph_cleanup_matches['rid'].max())
    
    return max_id + 1


def filter_pairs_df(pairs_df, threshold):

    # Filter out the pairs under the probability threshold
    pairs_df = pairs_df[pairs_df['prob'] >= threshold]

    return pairs_df

def filter_ground_truth_pairs_df(pairs_df):
    # Filter out the negative pairs of the ground truth
    pairs_df = pairs_df[pairs_df['label'] == 1]

    return pairs_df

def add_transitive_pairs(df):
    # Some datasets are not transitively consistent, so we add the missing transitive pairs of the ground truth record groups
    # We do this by adding the transitive closure of the ground truth pairs

    # First we create a graph from the ground truth pairs
    G = nx.from_pandas_edgelist(df, 'lid', 'rid')
    subgraphs = list(nx.connected_components(G))
    for c in subgraphs:
        if len(c) > 2:
            for i in c:
                for j in c:
                    if i != j:
                        if not df[(df['lid'] == i) & (df['rid'] == j)].empty:
                            continue
                        elif not df[(df['lid'] == j) & (df['rid'] == i)].empty:
                            continue
                        df = df.append({'lid': i, 'rid': j, 'label': 1}, ignore_index=True)

    return df

def transform_ids_to_positional_ids(df, dataset_name):
    # First load the test entity data file of the dataset
    test_entity_data_path = os.path.join('data', 'processed', dataset_name, 'test_entity_data.csv')
    test_entity_data = pd.read_csv(test_entity_data_path)
    test_entity_data['index'] = test_entity_data.index

    # Substitute lids and rids with the positional ids in the test entity data

    df = df.merge(test_entity_data[['index', 'id']], left_on='lid', right_on='id', how='left')
    df = df.rename(columns={'index': 'lid_pos'})
    df = df.drop(columns=['id'])

    df = df.merge(test_entity_data[['index', 'id']], left_on='rid', right_on='id', how='left')
    df = df.rename(columns={'index': 'rid_pos'})
    df = df.drop(columns=['id'])

    # Rename the positional ids to lids and rids
    df = df.drop(columns=['lid', 'rid'])
    df = df.rename(columns={'lid_pos': 'lid', 'rid_pos': 'rid'})

    return df 

def construct_sparse_matrix(df, max_id):
    sparse = coo_matrix(
        (
            np.ones(df.shape[0]),
            (df['lid'].values, df['rid'].values)
        ),
        shape=(max_id, max_id))
    
    # Make the matrix symmetric and all 1s
    sparse = sparse + sparse.T
    sparse.data = np.ones(sparse.data.shape[0])

    # Keep only the upper triangular part of the matrix (we do this to remove the duplicated pairs (i.e (A,B) and (B,A) pairs as well as the diagonal)
    sparse = sp.triu(sparse, k=1)

    return sparse



def get_true_and_false_positives(ground_truth_sparse, predictions_sparse):
    """
    Calculate True and False positives of the model predictions, to do this we iterate over the nonzero values of
    predictions_sparse.

    :param: ground_truth_sparse: The complete set of positive ground truth pairs in sparse format.
    :param: predictions_sparse: The complete set of positive pairs predicted by the model in sparse format.
    :return: true_positives, false_positives
    """
    ground_truth_sparse_csr = ground_truth_sparse.tocsr()
    rows, cols = predictions_sparse.nonzero()
    true_positives = set()
    false_positives = set()
    for row, col in zip(rows, cols):
        if ground_truth_sparse_csr[row, col] == 1 or ground_truth_sparse_csr[col,row] == 1: # We check both (A,B) and (B,A) in the ground truth
            true_positives.add((row, col))
        else:
            false_positives.add((row, col))

    return true_positives, false_positives

def get_false_negatives(ground_truth_sparse, predictions_sparse):
    """
    Calculate False Negatives of the model predictions, to do this we iterate over the nonzero values of
    ground_truth_sparse.

    :param: ground_truth_sparse: The complete set of positive ground truth pairs in sparse format.
    :param: all_predictions_sparse: The complete set of positive pairs predicted by the model in sparse format.
    :return: false_negatives

    """
    all_predictions_csr = predictions_sparse.tocsr()
    rows,cols = ground_truth_sparse.nonzero()
    false_negatives = set()
    for row,col in zip(rows,cols):
        if all_predictions_csr[row,col] == 0 and all_predictions_csr[col, row] == 0: # We check both (A,B) and (B,A) in the predictions
            false_negatives.add((row,col))

    return false_negatives

def get_pairwise_scores(true_positives, false_positives, false_negatives, big_subgraph_sizes = [], pairwise_pre_cleanup_scores = False):

    if pairwise_pre_cleanup_scores:
        false_positives = len(false_positives) + int(sum([subgraph_size * (subgraph_size - 1) / 2 for subgraph_size in big_subgraph_sizes]))
    else:
        false_positives = len(false_positives)

    precision = round(100 * len(true_positives) / (len(true_positives) + false_positives), 2)
    recall = round(100 * len(true_positives) / (len(true_positives) + len(false_negatives)), 2)
    f1_score = round((2 * precision * recall) /(precision + recall), 2)

    return precision, recall, f1_score


def get_subgraph_purity(matches_graph, ground_truth_sparse):

    subgraphs = list(nx.connected_components(matches_graph))

    # For each subgraph, calculate the % of true edges (including transitive edges)

    subgraph_purities = []
    ground_truth_sparse_csr = ground_truth_sparse.tocsr()

    for subgraph_idx, c in tqdm(enumerate(subgraphs), total=len(subgraphs), desc='Calculating subgraph purity'):
        # Get the edges of the subgraph from all_predictions_sparse
        subgraph_nodes = list(c)
        if len(subgraph_nodes) == 1:
            continue # Skip subgraphs with only one node
        purity = 0
        # Check how many of the edges of the subgraph are in the ground truth
        true_edges = ground_truth_sparse_csr[subgraph_nodes, :][:, subgraph_nodes].sum()
        purity += true_edges
        # Calculate the total number of edges in the complete subgraph
        number_of_edges = len(subgraph_nodes) * (len(subgraph_nodes) - 1) / 2
        purity = purity / number_of_edges
        # We keep track of the subgraph purity and its size
        subgraph_purities.append((purity, len(subgraph_nodes)))

    # To calculate overall subgraph purity, we weight each subgraph purity by its size and divide by the total number
    # of nodes

    subgraph_purity = sum([purity * size for purity, size in subgraph_purities]) / sum([size for purity, size in
                                                                                        subgraph_purities])
    return subgraph_purity

def update_all_scores_dict(all_scores_dict, scores_dict):

    all_scores_dict['recall_candidate_pairs'].append(scores_dict['recall_candidate_pairs'])

    all_scores_dict['pairwise_preds']['precision'].append(scores_dict['pairwise_preds']['precision'])
    all_scores_dict['pairwise_preds']['recall'].append(scores_dict['pairwise_preds']['recall'])
    all_scores_dict['pairwise_preds']['f1_score'].append(scores_dict['pairwise_preds']['f1_score'])
    all_scores_dict['pairwise_preds']['true_positives'].append(scores_dict['pairwise_preds']['true_positives'])
    all_scores_dict['pairwise_preds']['false_positives'].append(scores_dict['pairwise_preds']['false_positives'])
    all_scores_dict['pairwise_preds']['false_negatives'].append(scores_dict['pairwise_preds']['false_negatives'])

    all_scores_dict['pre_cleanup_matches']['precision'].append(scores_dict['pre_cleanup_matches']['precision'])
    all_scores_dict['pre_cleanup_matches']['recall'].append(scores_dict['pre_cleanup_matches']['recall'])
    all_scores_dict['pre_cleanup_matches']['f1_score'].append(scores_dict['pre_cleanup_matches']['f1_score'])

    all_scores_dict['post_graph_cleanup_matches']['precision'].append(scores_dict['post_graph_cleanup_matches']['precision'])
    all_scores_dict['post_graph_cleanup_matches']['recall'].append(scores_dict['post_graph_cleanup_matches']['recall'])
    all_scores_dict['post_graph_cleanup_matches']['f1_score'].append(scores_dict['post_graph_cleanup_matches']['f1_score'])

    try: 
        all_scores_dict['removed_edges']['pre_cleanup_true_positives'].append(scores_dict['removed_edges']['pre_cleanup_true_positives'])
        all_scores_dict['removed_edges']['pre_cleanup_false_positives'].append(scores_dict['removed_edges']['pre_cleanup_false_positives'])
        all_scores_dict['post_cleanup_true_positives'].append(scores_dict['post_cleanup_true_positives'])
        all_scores_dict['post_cleanup_false_positives'].append(scores_dict['post_cleanup_false_positives'])
        all_scores_dict['removed_edges']['true_positives_change'].append(scores_dict['removed_edges']['true_positives_change'])
        all_scores_dict['removed_edges']['false_positives_change'].append(scores_dict['removed_edges']['false_positives_change'])
    except KeyError:
        pass

    all_scores_dict['subgraph_purity']['pre-cleanup'].append(scores_dict['subgraph_purity']['pre-cleanup'])
    all_scores_dict['subgraph_purity']['post-cleanup'].append(scores_dict['subgraph_purity']['post-cleanup'])

    all_scores_dict['number_of_subgraphs']['pre-cleanup'].append(scores_dict['number_of_subgraphs']['pre-cleanup'])
    all_scores_dict['number_of_subgraphs']['post-cleanup'].append(scores_dict['number_of_subgraphs']['post-cleanup'])

    return all_scores_dict

def print_scores_for_each_labeling_budget(all_scores_dict, labeling_budgets_list):
    for i, labeling_budget in enumerate(labeling_budgets_list):
        print('Scores for labeling budget: ' + str(labeling_budget))
        print('-' * 80)
        print('PAIRWISE PREDS')
        print('-' * 80)
        print('Recall Candidate Pairs: ' + str(round(100 * all_scores_dict['recall_candidate_pairs'][0], 2)))
        print('True positives: ' + str(all_scores_dict['pairwise_preds']['true_positives'][i]))
        print('False positives: ' + str(all_scores_dict['pairwise_preds']['false_positives'][i]))
        print('False negatives: ' + str(all_scores_dict['pairwise_preds']['false_negatives'][i]))
        print('Precision: ' + str(all_scores_dict['pairwise_preds']['precision'][i]))
        print('Recall: ' + str(all_scores_dict['pairwise_preds']['recall'][i]))
        print('F1 score: ' + str(all_scores_dict['pairwise_preds']['f1_score'][i]))

        print('-' * 80)
        print('PRE CLEANUP MATCHES (PAIRWISE + TRANSITIVE)')
        print('-' * 80)
        print('Precision: ' + str(all_scores_dict['pre_cleanup_matches']['precision'][i])) 
        print('Recall: ' + str(all_scores_dict['pre_cleanup_matches']['recall'][i]))
        print('F1 score: ' + str(all_scores_dict['pre_cleanup_matches']['f1_score'][i]))

        print('-' * 80)
        print('POST GRAPH CLEANUP MATCHES')
        print('-' * 80)
        print('Precision: ' + str(all_scores_dict['post_graph_cleanup_matches']['precision'][i]))
        print('Recall: ' + str(all_scores_dict['post_graph_cleanup_matches']['recall'][i]))
        print('F1 score: ' + str(all_scores_dict['post_graph_cleanup_matches']['f1_score'][i]))
        try:
            print('True Positives Removed % : ' + str(all_scores_dict['removed_edges']['true_positives_change'][i]))
            print('False Positives Removed % : ' + str(all_scores_dict['removed_edges']['false_positives_change'][i]))
        except KeyError:
            pass

        print('-' * 80)
        print('SUBGRAPH PURITY')
        print('-' * 80)
        print('Subgraph purity pre-cleanup: ' + str(all_scores_dict['subgraph_purity']['pre-cleanup'][i]))
        print('Subgraph purity post-cleanup: ' + str(all_scores_dict['subgraph_purity']['post-cleanup'][i]))

        print('-' * 80)
        print('NUMBER OF SUBGRAPHS')
        print('-' * 80)
        print('Number of subgraphs pre-cleanup: ' + str(all_scores_dict['number_of_subgraphs']['pre-cleanup'][i]))
        print('Number of subgraphs post-cleanup: ' + str(all_scores_dict['number_of_subgraphs']['post-cleanup'][i]))


def plot_labeling_budget_vs_f1_score(all_scores_dict, exp_args, plot_final_path, training_budgets_CLER = None, CLER_results_dict = None, post_cleanup_CLER_scores_dict = None):
    # Plot the labeling budget vs f1 score graph
    cleanup_labeling_budgets = [int(lb) for lb in exp_args.labeling_budgets_list]
    exp_args.labeling_budgets_list = [int(lb) + 10000 for lb in exp_args.labeling_budgets_list] # Add the 10k training budget of the benchmark method (DistilBERT 10k)
    if training_budgets_CLER is not None:
        training_budgets_CLER = np.array(training_budgets_CLER)
    all_scores_dict['post_graph_cleanup_matches']['f1_score'] = [f1/100 for f1 in all_scores_dict['post_graph_cleanup_matches']['f1_score']]
    all_scores_dict['post_graph_cleanup_matches']['precision'] = [precision/100 for precision in all_scores_dict['post_graph_cleanup_matches']['precision']]
    all_scores_dict['post_graph_cleanup_matches']['recall'] = [recall/100 for recall in all_scores_dict['post_graph_cleanup_matches']['recall']]

    # Make a figure with 3 subplots
    fig, ax = plt.subplots(3, 1)
    # Set the size of the figures 
    fig.set_size_inches(1.25*11.69,1.25*8.27)
    fig.suptitle('Labeling budgets vs Precision, Recall and F1 scores')

    labeling_budgets_benchmark = np.array(exp_args.labeling_budgets_list) - np.array(all_scores_dict['unspent_labeling_budgets'])

    ax[0].plot(labeling_budgets_benchmark, all_scores_dict['post_graph_cleanup_matches']['precision'], label='DistilBERT 10k', marker='o', color='orange', linewidth=2)
    if training_budgets_CLER is not None and CLER_results_dict is not None:
        ax[0].plot(training_budgets_CLER, CLER_results_dict['precision_scores'], label='CLER Pre-TransClean', marker='o', color='r', linewidth = 2)
        if post_cleanup_CLER_scores_dict is not None:
            # Plot the post cleanup precision scores of the CLER models
            for key, experiment_dict in post_cleanup_CLER_scores_dict.items():
                CLER_labeling_budget = key.split('_')[-1]
                labeling_budgets = [int(lb) + int(CLER_labeling_budget) for lb in experiment_dict['labeling_budgets']]
                labeling_budgets = np.array(labeling_budgets) - np.array(experiment_dict['unspent_labeling_budgets'])
                ax[0].plot(labeling_budgets, experiment_dict['precision_scores'], label='CLER ' + CLER_labeling_budget, marker='o', linewidth=2)
    ax[0].set_title('Precision scores')
    ax[0].legend(loc='lower right')
    ax[0].set(ylabel='Precision')

    ax[1].plot(labeling_budgets_benchmark, all_scores_dict['post_graph_cleanup_matches']['recall'], label='DistilBERT 10k', marker='o', color='orange', linewidth=2)
    if training_budgets_CLER is not None and CLER_results_dict is not None:
        ax[1].plot(training_budgets_CLER, CLER_results_dict['recall_scores'], label='CLER Pre-TransClean', marker='o', color='r', linewidth = 2)
        if post_cleanup_CLER_scores_dict is not None:
            # Plot the post cleanup recall scores of the CLER models
            for key, experiment_dict in post_cleanup_CLER_scores_dict.items():
                CLER_labeling_budget = key.split('_')[-1]
                labeling_budgets = [int(lb) + int(CLER_labeling_budget) for lb in experiment_dict['labeling_budgets']]
                labeling_budgets = np.array(labeling_budgets) - np.array(experiment_dict['unspent_labeling_budgets'])
                ax[1].plot(labeling_budgets, experiment_dict['recall_scores'], label='CLER ' + CLER_labeling_budget, marker='o', linewidth=2)
    ax[1].set_title('Recall scores')
    ax[1].legend(loc='upper right')
    ax[1].set(ylabel='Recall')

    ax[2].plot(labeling_budgets_benchmark, all_scores_dict['post_graph_cleanup_matches']['f1_score'], label='DistilBERT 10k', marker='o', color='orange', linewidth=2)
    if training_budgets_CLER is not None and CLER_results_dict is not None:
        ax[2].plot(training_budgets_CLER, CLER_results_dict['f1_scores'], label='CLER Pre-TransClean', marker='o', color='r', linewidth = 2)
        if post_cleanup_CLER_scores_dict is not None:
            # Plot the post cleanup f1 scores of the CLER models
            for key, experiment_dict in post_cleanup_CLER_scores_dict.items():
                CLER_labeling_budget = key.split('_')[-1]
                labeling_budgets = [int(lb) + int(CLER_labeling_budget) for lb in experiment_dict['labeling_budgets']] 
                labeling_budgets = np.array(labeling_budgets) - np.array(experiment_dict['unspent_labeling_budgets'])
                # Change the label to CLER + the labeling budget
                key = 'CLER ' + CLER_labeling_budget
                ax[2].plot(labeling_budgets, experiment_dict['f1_scores'], label=key, marker='o', linewidth=2)
    ax[2].set_title('F1 scores')
    ax[2].legend(loc='upper right')
    ax[2].set(ylabel='F1 score')
    
    # Tilt the x-axis labels
    plt.xticks(rotation=22.5)
    # Add the labeling budget to each point as a text above the point
    for ax_idx in range(3):
        for i, txt in enumerate(cleanup_labeling_budgets):
            if ax_idx == 0:
                ax[ax_idx].annotate(txt, (labeling_budgets_benchmark[i] + 500, all_scores_dict['post_graph_cleanup_matches']['precision'][i] - 0.01))
                if training_budgets_CLER is not None:
                    ax[ax_idx].annotate(txt, (training_budgets_CLER[i] + 500, CLER_results_dict['precision_scores'][i] - 0.01))
                    if post_cleanup_CLER_scores_dict is not None:
                        for key, experiment_dict in post_cleanup_CLER_scores_dict.items():
                            CLER_labeling_budget = key.split('_')[-1]
                            labeling_budgets = [int(lb) + int(CLER_labeling_budget) for lb in experiment_dict['labeling_budgets']]
                            labeling_budgets = np.array(labeling_budgets) - np.array(experiment_dict['unspent_labeling_budgets'])
                            ax[ax_idx].annotate(txt, (labeling_budgets[i] - 500, experiment_dict['precision_scores'][i] - 0.06), ha='left', rotation=-60)
            elif ax_idx == 1:
                ax[ax_idx].annotate(txt, (labeling_budgets_benchmark[i] + 500, all_scores_dict['post_graph_cleanup_matches']['recall'][i] - 0.006))
                if training_budgets_CLER is not None:
                    ax[ax_idx].annotate(txt, (training_budgets_CLER[i] + 500, CLER_results_dict['recall_scores'][i] - 0.006))
                    if post_cleanup_CLER_scores_dict is not None:
                        for key, experiment_dict in post_cleanup_CLER_scores_dict.items():
                            CLER_labeling_budget = key.split('_')[-1]
                            labeling_budgets = [int(lb) + int(CLER_labeling_budget) for lb in experiment_dict['labeling_budgets']]
                            labeling_budgets = np.array(labeling_budgets) - np.array(experiment_dict['unspent_labeling_budgets'])
                            ax[ax_idx].annotate(txt, (labeling_budgets[i] + 500, experiment_dict['recall_scores'][i] - 0.04), ha='left', rotation=-45)
            elif ax_idx == 2:
                ax[ax_idx].annotate(txt, (labeling_budgets_benchmark[i] + 500, all_scores_dict['post_graph_cleanup_matches']['f1_score'][i]- 0.006))
                if training_budgets_CLER is not None:
                    ax[ax_idx].annotate(txt, (training_budgets_CLER[i] + 500, CLER_results_dict['f1_scores'][i] - 0.006))
                    if post_cleanup_CLER_scores_dict is not None:
                        for key, experiment_dict in post_cleanup_CLER_scores_dict.items():
                            CLER_labeling_budget = key.split('_')[-1]
                            labeling_budgets = [int(lb) + int(CLER_labeling_budget) for lb in experiment_dict['labeling_budgets']] 
                            labeling_budgets = np.array(labeling_budgets) - np.array(experiment_dict['unspent_labeling_budgets'])
                            ax[ax_idx].annotate(txt, (labeling_budgets[i] + 500, experiment_dict['f1_scores'][i] - 0.04), ha='left', rotation=-45)
                        
    # Tilt the x-axis labels
    plt.xticks(rotation=22.5)
    plt.xlabel('Total Labeling Effort (Training + Transitivity Cleanup)')
    fig.tight_layout()
    plt.savefig(os.path.join(plot_final_path, 'labeling_budget_vs_scores.png'), bbox_inches='tight')

if __name__ == '__main__':
    exp_args = get_scores_args()


    all_scores_dict = {}        
    all_scores_dict['recall_candidate_pairs'] = []
    all_scores_dict['pairwise_preds'] = {'precision': [], 'recall': [], 'f1_score': [], 'true_positives': [], 'false_positives': [], 'false_negatives': []}
    all_scores_dict['pre_cleanup_matches'] = {'precision': [], 'recall': [], 'f1_score': [], 'true_positives': [], 'false_positives': [], 'false_negatives': []}
    all_scores_dict['post_graph_cleanup_matches'] = {'precision': [], 'recall': [], 'f1_score': [], 'true_positives': [], 'false_positives': [], 'false_negatives': []}
    all_scores_dict['removed_edges'] = {'pre_cleanup_true_positives': [], 'pre_cleanup_false_positives': [], 'true_positives_change': [], 'false_positives_change': []}
    all_scores_dict['post_cleanup_true_positives'] = []
    all_scores_dict['post_cleanup_false_positives'] = []
    all_scores_dict['subgraph_purity'] = {'pre-cleanup': [], 'post-cleanup': []}
    all_scores_dict['number_of_subgraphs'] = {'pre-cleanup': [], 'post-cleanup': []}
    all_scores_dict['unspent_labeling_budgets'] = []

    if exp_args.labeling_budgets_list is not None:
        # We are calculating the scores for a labeling budget vs f1 score graph, we do this only for a single experiment
        assert len(exp_args.experiment_names_list) == 1

    
    if exp_args.labeling_budgets_list is None:

        for i, experiment_name in enumerate(exp_args.experiment_names_list):
            matching_folder_path = dataset_results_folder_path__with_subfolders(subfolder_list=[exp_args.dataset_name, experiment_name])

            if not os.path.exists(os.path.join(matching_folder_path, 'scores_dict.pkl')):

                if exp_args.post_edge_recovery:
                    scores_dict = get_scores(matching_folder_path, exp_args.ground_truth_path, exp_args.dataset_name, exp_args.threshold, 
                                            exp_args.non_positional_ids,
                                            post_edge_recovery = True)            
                else:
                    scores_dict = get_scores(matching_folder_path, exp_args.ground_truth_path, exp_args.dataset_name, exp_args.threshold, exp_args.non_positional_ids)

                with open(os.path.join(matching_folder_path, 'scores_dict.pkl'), 'wb') as f:
                    pickle.dump(scores_dict, f)

            else:
                with open(os.path.join(matching_folder_path, 'scores_dict.pkl'), 'rb') as f:
                    scores_dict = pickle.load(f)

            all_scores_dict = update_all_scores_dict(all_scores_dict, scores_dict)
    
    else:
        # We are calculating the scores for a labeling budget vs f1 score graph
        for i, labeling_budget in enumerate(exp_args.labeling_budgets_list):
            matching_folder_path = dataset_results_folder_path__with_subfolders(subfolder_list=[exp_args.dataset_name, exp_args.experiment_names_list[0]])
            if os.path.exists(os.path.join(matching_folder_path, 'labeling_budget_{}'.format(labeling_budget), 'scores_dict.pkl')):
                with open(os.path.join(matching_folder_path, 'labeling_budget_{}'.format(labeling_budget), 'scores_dict.pkl'), 'rb') as f:
                    scores_dict = pickle.load(f)
            else:
                scores_dict = get_scores(matching_folder_path, exp_args.ground_truth_path, exp_args.dataset_name, exp_args.threshold, exp_args.non_positional_ids, post_edge_recovery = True, labeling_budget = labeling_budget)
                with open(os.path.join(matching_folder_path, 'labeling_budget_{}'.format(labeling_budget), 'scores_dict.pkl'), 'wb') as f:
                    pickle.dump(scores_dict, f)
            all_scores_dict = update_all_scores_dict(all_scores_dict, scores_dict)
            # Add the unspent labeling budget to the scores_dict
            post_edge_recovery_path = os.path.join(matching_folder_path, 'labeling_budget_{}'.format(labeling_budget), 'post_edge_recovery')
            post_edge_recovery_df = pd.read_csv(os.path.join(post_edge_recovery_path, 'post_edge_recovery.csv'))
            all_scores_dict['unspent_labeling_budgets'].append(post_edge_recovery_df['labeling_budget_left'].values[0])


    if exp_args.labeling_budgets_list is None:

        # Print the true positives, false positives and false negatives for the pairwise predictions and the mean and std of all scores


        print('Scores for experiments: ' + str(exp_args.experiment_names_list) + ' on dataset: ' + str(exp_args.dataset_name))
        print('-' * 80)
        print('PAIRWISE PREDS')
        print('-' * 80)
        print('Recall Candidate Pairs: ' + str(round(100 * all_scores_dict['recall_candidate_pairs'][0], 2)))
        print('True positives: ' + str(np.mean(all_scores_dict['pairwise_preds']['true_positives'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['true_positives'])))
        print('False positives: ' + str(np.mean(all_scores_dict['pairwise_preds']['false_positives'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['false_positives'])))
        print('False negatives: ' + str(np.mean(all_scores_dict['pairwise_preds']['false_negatives'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['false_negatives'])))
        print('Precision: ' + str(np.mean(all_scores_dict['pairwise_preds']['precision'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['precision'])))
        print('Recall: ' + str(np.mean(all_scores_dict['pairwise_preds']['recall'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['recall'])))
        print('F1 score: ' + str(np.mean(all_scores_dict['pairwise_preds']['f1_score'])) + ' +/- ' + str(np.std(all_scores_dict['pairwise_preds']['f1_score'])))

        print('-' * 80)
        print('PRE CLEANUP MATCHES (PAIRWISE + TRANSITIVE)')
        print('-' * 80)
        print('Precision: ' + str(np.mean(all_scores_dict['pre_cleanup_matches']['precision'])) + ' +/- ' + str(np.std(all_scores_dict['pre_cleanup_matches']['precision'])))
        print('Recall: ' + str(np.mean(all_scores_dict['pre_cleanup_matches']['recall'])) + ' +/- ' + str(np.std(all_scores_dict['pre_cleanup_matches']['recall'])))
        print('F1 score: ' + str(np.mean(all_scores_dict['pre_cleanup_matches']['f1_score'])) + ' +/- ' + str(np.std(all_scores_dict['pre_cleanup_matches']['f1_score'])))

        print('-' * 80)
        print('POST GRAPH CLEANUP MATCHES')
        print('-' * 80)
        print('Precision: ' + str(np.mean(all_scores_dict['post_graph_cleanup_matches']['precision'])) + ' +/- ' + str(np.std(all_scores_dict['post_graph_cleanup_matches']['precision'])))
        print('Recall: ' + str(np.mean(all_scores_dict['post_graph_cleanup_matches']['recall'])) + ' +/- ' + str(np.std(all_scores_dict['post_graph_cleanup_matches']['recall'])))
        print('F1 score: ' + str(np.mean(all_scores_dict['post_graph_cleanup_matches']['f1_score'])) + ' +/- ' + str(np.std(all_scores_dict['post_graph_cleanup_matches']['f1_score'])))
        try:
            print('True Positives Removed %: ' + str(np.mean(all_scores_dict['removed_edges']['true_positives_change'])) + ' +/- ' + str(np.std(all_scores_dict['removed_edges']['true_positives_change'])))
            print('False Positives Removed %: ' + str(np.mean(all_scores_dict['removed_edges']['false_positives_change'])) + ' +/- ' + str(np.std(all_scores_dict['removed_edges']['false_positives_change'])))
        except KeyError:
            pass

        print('-' * 80)
        print('SUBGRAPH PURITY')
        print('-' * 80)
        print('Subgraph purity pre-cleanup: ' + str(np.mean(all_scores_dict['subgraph_purity']['pre-cleanup'])) + ' +/- ' + str(np.std(all_scores_dict['subgraph_purity']['pre-cleanup'])))
        print('Subgraph purity post-cleanup: ' + str(np.mean(all_scores_dict['subgraph_purity']['post-cleanup'])) + ' +/- ' + str(np.std(all_scores_dict['subgraph_purity']['post-cleanup'])))

        print('-' * 80)
        print('NUMBER OF SUBGRAPHS')
        print('-' * 80)
        print('Number of subgraphs pre-cleanup: ' + str(np.mean(all_scores_dict['number_of_subgraphs']['pre-cleanup'])) + ' +/- ' + str(np.std(all_scores_dict['number_of_subgraphs']['pre-cleanup'])))
        print('Number of subgraphs post-cleanup: ' + str(np.mean(all_scores_dict['number_of_subgraphs']['post-cleanup'])) + ' +/- ' + str(np.std(all_scores_dict['number_of_subgraphs']['post-cleanup'])))
    
    else:
        print_scores_for_each_labeling_budget(all_scores_dict, exp_args.labeling_budgets_list)
        plot_final_path = dataset_results_folder_path__with_subfolders(subfolder_list=[exp_args.dataset_name, exp_args.experiment_names_list[0]])
        if exp_args.add_CLER_scores:
            # Recover the CLER scores ( they can be found in all the results folders of the CLER experiments, ending in _CLER_{} where {} is the labeling budget)
            matching_folder_path = dataset_results_folder_path__with_subfolders(subfolder_list=[exp_args.dataset_name])
            cler_experiments_folders_list = [folder for folder in os.listdir(matching_folder_path) if 'CLER' in folder and folder.split('_')[-1].isdigit()]
            labeling_budgets_list = [int(folder.split('_')[-1]) for folder in cler_experiments_folders_list]
        
            # Sort the folders list by labeling budget
            cler_results_folders_list = [x for _, x in sorted(zip(labeling_budgets_list, cler_experiments_folders_list))]
            CLER_results_dict = {}
            CLER_results_dict['precision_scores'] = []
            CLER_results_dict['recall_scores'] = []
            CLER_results_dict['f1_scores'] = []
            for results_folder in cler_results_folders_list:
                cler_results_folder_path = os.path.join(matching_folder_path, results_folder)
                cler_metrics = pd.read_csv(os.path.join(cler_results_folder_path, 'CLER_metrics.csv'))
                CLER_results_dict['precision_scores'].append(cler_metrics['post_graph_cleanup_matches_precision'].values[0])
                CLER_results_dict['recall_scores'].append(cler_metrics['post_graph_cleanup_matches_recall'].values[0])
                CLER_results_dict['f1_scores'].append(cler_metrics['post_graph_cleanup_matches_f1'].values[0])

            if exp_args.CLER_post_cleanup_list:
                # Add the post-finetuning cleanup CLER scores to the labeling budget vs f1 score graph
                post_cleanup_CLER_scores_dict = {}
                for CLER_experiment in exp_args.CLER_post_cleanup_list:
                    post_cleanup_CLER_scores_dict[CLER_experiment] = {}
                    CLER_experiment_labeling_budgets = [int(folder.split('_')[-1]) for folder in os.listdir(os.path.join(matching_folder_path, CLER_experiment)) if folder.split('_')[-1].isdigit()]
                    CLER_experiment_labeling_budgets = sorted(CLER_experiment_labeling_budgets)
                    post_cleanup_CLER_scores_dict[CLER_experiment]['labeling_budgets'] = CLER_experiment_labeling_budgets
                    post_cleanup_CLER_scores_dict[CLER_experiment]['precision_scores'] = []
                    post_cleanup_CLER_scores_dict[CLER_experiment]['recall_scores'] = []
                    post_cleanup_CLER_scores_dict[CLER_experiment]['f1_scores'] = []
                    post_cleanup_CLER_scores_dict[CLER_experiment]['unspent_labeling_budgets'] = []
                    for labeling_budget in CLER_experiment_labeling_budgets:
                        if not os.path.exists(os.path.join(matching_folder_path, CLER_experiment, 'labeling_budget_{}'.format(labeling_budget), 'scores_dict.pkl')):
                            scores_dict = get_scores(os.path.join(matching_folder_path, CLER_experiment), 
                                                     exp_args.ground_truth_path, exp_args.dataset_name, exp_args.threshold, exp_args.non_positional_ids, post_edge_recovery = True, labeling_budget = labeling_budget)
                            with open(os.path.join(matching_folder_path, CLER_experiment, 'labeling_budget_{}'.format(labeling_budget), 'scores_dict.pkl'), 'wb') as f:
                                pickle.dump(scores_dict, f)
                        else:
                            with open(os.path.join(matching_folder_path, CLER_experiment, 'labeling_budget_{}'.format(labeling_budget), 'scores_dict.pkl'), 'rb') as f:
                                scores_dict = pickle.load(f)
                        # Add the post cleanup scores to the dictionary
                        post_cleanup_CLER_scores_dict[CLER_experiment]['precision_scores'].append(scores_dict['post_graph_cleanup_matches']['precision']/100)
                        post_cleanup_CLER_scores_dict[CLER_experiment]['recall_scores'].append(scores_dict['post_graph_cleanup_matches']['recall']/100)
                        post_cleanup_CLER_scores_dict[CLER_experiment]['f1_scores'].append(scores_dict['post_graph_cleanup_matches']['f1_score']/100)
                        # Recover also the unspent labeling budget from the post_edge_recovery folder
                        post_edge_recovery_folder_path = os.path.join(matching_folder_path, CLER_experiment, 'labeling_budget_{}'.format(labeling_budget), 'post_edge_recovery')
                        post_edge_recovery_df = pd.read_csv(os.path.join(post_edge_recovery_folder_path, 'post_edge_recovery.csv'))
                        unspent_labeling_budget = post_edge_recovery_df['labeling_budget_left'].values[0]
                        # Record the unspent labeling budget in the post_cleanup_CLER_scores_dict
                        post_cleanup_CLER_scores_dict[CLER_experiment]['unspent_labeling_budgets'].append(unspent_labeling_budget)
                        

                
                plot_labeling_budget_vs_f1_score(all_scores_dict, exp_args, plot_final_path, sorted(labeling_budgets_list), CLER_results_dict, 
                                                 post_cleanup_CLER_scores_dict)

            else:
            
                plot_labeling_budget_vs_f1_score(all_scores_dict, exp_args, plot_final_path, sorted(labeling_budgets_list), CLER_results_dict)



        else:
            plot_labeling_budget_vs_f1_score(all_scores_dict, exp_args, plot_final_path)
