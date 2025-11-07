import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.getcwd())

from src.helpers.path_helper import *



def get_visualization_args():
    parser = argparse.ArgumentParser(description='Visualize matching with finetuning cleanup, post-finetuning cleanup/checks & edge recovery')

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--labeling_budgets_list', type=int, action='append', required=True)
    parser.add_argument('--add_transitive_preds', action='store_true', default=False)
    args = parser.parse_args()

    return args

def load_matching_results(matching_folder_path):
    """
    For each matching experiment we load the following result files:
    - finetune_cleanup_dict.csv from the last finetuning iteration
    - post_finetuning_cleanup.csv
    - post_finetuning_check.csv
    - post_edge_recovery.csv
    """
    # Get the finetune_cleanup_dict.csv, to do this we need to find out what the last finetuning iteration was
    finetune_cleanup_dict_folders = os.listdir(matching_folder_path)
    # Get all the folders called finetune_iteration_{} and sort them
    finetune_cleanup_dict_folders = [folder for folder in finetune_cleanup_dict_folders if 'finetune_iteration' in folder]
    highest_finetune_iteration = max([int(folder.split('_')[-1]) for folder in finetune_cleanup_dict_folders])
    finetune_cleanup_dict_folder = 'finetune_iteration_{}'.format(highest_finetune_iteration)
    finetune_cleanup_dict_path = os.path.join(matching_folder_path, finetune_cleanup_dict_folder, 'finetune_cleanup_dict.csv')
    finetune_cleanup_dict = pd.read_csv(finetune_cleanup_dict_path)

    # Get the post_finetuning_cleanup.csv
    post_finetuning_cleanup_path = os.path.join(matching_folder_path, 'post_finetuning_cleanup')
    post_finetuning_cleanup = pd.read_csv(os.path.join(post_finetuning_cleanup_path, 'post_finetuning_cleanup.csv'))

    # Get the post_finetuning_check.csv
    post_finetuning_check_path = os.path.join(matching_folder_path, 'post_finetuning_check')
    post_finetuning_check = pd.read_csv(os.path.join(post_finetuning_check_path, 'post_finetuning_check.csv'))

    # Get the post_edge_recovery.csv
    post_edge_recovery_path = os.path.join(matching_folder_path, 'post_edge_recovery')
    post_edge_recovery = pd.read_csv(os.path.join(post_edge_recovery_path, 'post_edge_recovery.csv'))

    return finetune_cleanup_dict, post_finetuning_cleanup, post_finetuning_check, post_edge_recovery


def visualize_TransClean_cleanup(matching_folder_path, model_name, labeling_budget, add_transitive_preds, finetune_cleanup_dict, post_finetuning_cleanup, post_finetuning_check, post_edge_recovery):
    """
    Visualize the TransClean cleanup, post-finetuning cleanup/checks & edge recovery. We plot a Fig with 3 bar plots showed next to each other. 

    - 1) The number of False Positives
    - 2) The number of True Positives
    - 3) The labelling effort (number of manually labeled edges)

    These quantities are plotted for each finetuning iteration, post-finetuning cleanup/checks & edge recovery.
    """

    # Gather the False Positives, True Positives and labelling effort for each finetuning iteration
    false_positives_finetuning = finetune_cleanup_dict['false_positives']
    true_positives_finetuning = finetune_cleanup_dict['true_positives']
    labeled_pairs_finetuning = finetune_cleanup_dict['labeled_pairs']

    # Get the False Positives, True Positives and labelling effort for the post_finetuning_cleanup
    false_positives_post_finetuning_cleanup = post_finetuning_cleanup['starting_false_positives'] - post_finetuning_cleanup['removed_false_positives_cleanup']
    true_positives_post_finetuning_cleanup = post_finetuning_cleanup['starting_true_positives'] - post_finetuning_cleanup['removed_true_positives_cleanup']
    labeled_pairs_post_finetuning_cleanup = labeled_pairs_finetuning.iloc[-1]

    # Get the False Positives, True Positives and labelling effort for the post_finetuning_check
    false_positives_post_finetuning_check = post_finetuning_check['starting_false_positives'] - post_finetuning_check['removed_false_positives']
    true_positives_post_finetuning_check = post_finetuning_check['starting_true_positives']
    labeled_pairs_post_finetuning_check = labeled_pairs_post_finetuning_cleanup + post_finetuning_check['labeled_pairs_large_subgraphs_check'] + post_finetuning_check['labeled_pairs_full_subgraph_check_heuristic'] + post_finetuning_check['labeled_pairs_transitive_check_heuristic']

    # Get the False Positives, True Positives and labelling effort for the post_edge_recovery
    false_positives_post_edge_recovery = post_edge_recovery['post_edge_recovery_false_positives']
    true_positives_post_edge_recovery = post_edge_recovery['post_edge_recovery_true_positives']
    labeled_pairs_post_edge_recovery = labeling_budget - post_edge_recovery['labeling_budget_left']

    # Join the False Positives, True Positives and labelling effort for each step
    false_positives = np.concatenate([false_positives_finetuning, false_positives_post_finetuning_cleanup, false_positives_post_finetuning_check, false_positives_post_edge_recovery])
    true_positives = np.concatenate([true_positives_finetuning, true_positives_post_finetuning_cleanup, true_positives_post_finetuning_check, true_positives_post_edge_recovery])
    labeled_pairs = np.concatenate([labeled_pairs_finetuning, pd.Series(labeled_pairs_post_finetuning_cleanup), labeled_pairs_post_finetuning_check, labeled_pairs_post_edge_recovery])  

    if add_transitive_preds:
        positive_transitive_preds_finetuning = finetune_cleanup_dict['positive_transitive_preds']  
        negative_transitive_preds_finetuning = finetune_cleanup_dict['negative_transitive_preds']
        
        positive_transitive_preds_post_finetuning_cleanup = post_finetuning_cleanup['starting_positive_transitive_preds']
        negative_transitive_preds_post_finetuning_cleanup = post_finetuning_cleanup['starting_negative_transitive_preds']

        positive_transitive_preds_post_finetuning_check = post_finetuning_check['starting_positive_transitive_preds']
        negative_transitive_preds_post_finetuning_check = post_finetuning_check['starting_negative_transitive_preds']

        positive_transitive_preds_post_edge_recovery = post_edge_recovery['post_edge_recovery_positive_transitive_preds']
        negative_transitive_preds_post_edge_recovery = post_edge_recovery['post_edge_recovery_negative_transitive_preds']

        positive_transitive_preds = np.concatenate([positive_transitive_preds_finetuning, positive_transitive_preds_post_finetuning_cleanup, 
                                                    positive_transitive_preds_post_finetuning_check, positive_transitive_preds_post_edge_recovery])
        negative_transitive_preds = np.concatenate([negative_transitive_preds_finetuning, negative_transitive_preds_post_finetuning_cleanup, 
                                                    negative_transitive_preds_post_finetuning_check, negative_transitive_preds_post_edge_recovery])
    # Put all the data in a DataFrame
    df = pd.DataFrame({'False Positives': false_positives, 'True Positives': true_positives, 'Labeled Pairs': labeled_pairs})
    if add_transitive_preds:
        df['Positive Transitive Predictions'] = positive_transitive_preds
        df['Negative Transitive Predictions'] = negative_transitive_preds
    
    # Make a list for the x-axis labels
    x_labels = ['Finetuning Iteration {}'.format(i) for i in range(1, len(false_positives_finetuning) + 1)] + ['Post Finetuning Cleanup', 'Post Finetuning Check', 'Post Edge Recovery']

    df['idx'] = x_labels
    fig, ax1 = plt.subplots(figsize=(17.5, 10))
    tidy = df.melt(id_vars='idx').rename(columns=str.title)
    sns.barplot(x = 'Idx', y = 'Value', hue = 'Variable', data = tidy, ax=ax1)
    sns.despine(fig)

    # Plot at the top of each column the difference with the previous column of the same variable
    variables = ['False Positives', 'True Positives', 'Labeled Pairs']
    if add_transitive_preds:
        variables.append('Positive Transitive Predictions')
        variables.append('Negative Transitive Predictions')

    for i in range(1, len(x_labels)):
        for j, variable in enumerate(variables):
            diff = df[variable].iloc[i] - df[variable].iloc[i-1]
            if len(variables) == 3:
                if variable == 'False Positives':
                    # False Positives are plotted as the leftmost bar
                    x_pos = i - 0.3
                elif variable == 'True Positives':
                    # True Positives are plotted as the middle bar
                    x_pos = i
                elif variable == 'Labeled Pairs':
                    # Labeled Pairs are plotted as the rightmost bar
                    x_pos = i + 0.3
            if len(variables) == 5:
                if variable == 'False Positives':
                    x_pos = i - 0.4
                elif variable == 'True Positives':
                    x_pos = i - 0.2
                elif variable == 'Labeled Pairs':
                    x_pos = i 
                elif variable == 'Positive Transitive Predictions':
                    x_pos = i + 0.2
                elif variable == 'Negative Transitive Predictions':
                    x_pos = i + 0.4

            if diff > 0:
                ax1.text(x_pos, df[variable].iloc[i] + 0.15 * df[variable].iloc[i], '+{}'.format(diff), ha='center', va='top', rotation=0, fontsize=9, color='black')
            elif diff < 0:
                ax1.text(x_pos, df[variable].iloc[i] + 0.15 * df[variable].iloc[i], '-{}'.format(diff), ha='center', va='top', rotation=0, fontsize=9, color='black')
            else:
                ax1.text(x_pos, df[variable].iloc[i] + 0.15 * df[variable].iloc[i], '0', ha='center', va='top', rotation=0, fontsize=9, color='black')


    plt.title('Matching Process Visualization - {} with Labeling Budget {}'.format(model_name,labeling_budget), pad = 25)
    # Tilt the x-axis labels
    plt.xticks(rotation=22.5)
    # Make the y scale logarithmic
    plt.yscale('log')
    # Add 'log-scale' to the y-axis label
    plt.ylabel('Edge Counts (log-scale)')
    # Put the legend on the upper righ side of the Fig, outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(0.55, 1.05), ncol = 3, fancybox=True, shadow=True)
    # Add a grid to the plot
    plt.grid(axis='y')


    # Add a text box with the title "Final Quantities" and the final number of each variable (False Positives, True Positives, Labeled Pairs)
    props = dict(boxstyle='Square', facecolor='xkcd:light blue', alpha=0.5)
    textstr = '\n'.join(['{}: {}'.format(variable, df[variable].iloc[-1]) for variable in ['False Positives', 'True Positives', 'Labeled Pairs']])

    ax1.text(-0.1, 1.05, 'Final Quantities:', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.text(0.05, 1.10, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # Move the text box to the upper left corner
    
    plt.savefig(os.path.join(matching_folder_path, 'matching_process_visualization_labeling_budget_{}.png'.format(labeling_budget)))

if __name__ == '__main__':
    vis_args = get_visualization_args()
    experiment_name = vis_args.experiment_name

    for i, labeling_budget in enumerate(vis_args.labeling_budgets_list):
        matching_folder_path = dataset_results_folder_path__with_subfolders(subfolder_list=[vis_args.dataset_name, experiment_name])
        matching_folder_path = os.path.join(matching_folder_path, 'labeling_budget_{}'.format(labeling_budget))

        matching_results = load_matching_results(matching_folder_path)
        visualize_TransClean_cleanup(matching_folder_path, vis_args.model_name, labeling_budget, vis_args.add_transitive_preds, *matching_results)
        



