import wandb
import os
import argparse
import re
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull

from performance_indicators import hypervolume

import seaborn as sns


def plot_hypervolume_development(hypervolumes, output_folder):
    max_length = max(len(hv) for hv in hypervolumes)
    hypervolumes_array = np.array([np.pad(hv, (0, max_length - len(hv)), 'constant', constant_values=np.nan) for hv in hypervolumes], dtype=float)
    
    # Calculate mean and standard deviation ignoring NaNs
    mean_hv = np.nanmean(hypervolumes_array, axis=0)
    std_hv = np.nanstd(hypervolumes_array, axis=0)

    #plt.figure(figsize=(10, 6))

    iterations = np.arange(1, max_length + 1)
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.rcParams['figure.dpi'] = 360

    plt.plot(iterations, mean_hv, label='Mean Hypervolume', color='blue', linewidth=2, linestyle='--', marker='o', markersize=5, markerfacecolor='white')
    plt.fill_between(iterations, mean_hv - std_hv, mean_hv + std_hv, color='blue', alpha=0.2, label='Std Deviation')

    plt.xlabel('Evaluation Iteration', fontsize=10)
    plt.ylabel('Hypervolume', fontsize=10)
    plt.title('Hypervolume Development over iterations', fontsize=12, fontweight='bold')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    sns.despine(left=True)
    plt.tight_layout()

    output_path = os.path.join(output_folder, 'hypervolume_development.png')
    plt.savefig(output_path)
    plt.close()

def load_hypervolumes(input_file1, input_file2):
    with open(input_file1, "r") as f:
        data1 = json.load(f)
    with open(input_file2, "r") as f:
        data2 = json.load(f)
    
    hypervolumes1 = data1
    hypervolumes2 = data2

    return hypervolumes1, hypervolumes2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file1", help="Path to the first input file", type=str, required=True)
    parser.add_argument("--input_file2", help="Path to the second input file", type=str, required=True)
    parser.add_argument("--output_path", help="Path to file where the hypervolumes will be saved", type=str, required=True)

    args = parser.parse_args()

    hypervolume1, hypervolume2 = load_hypervolumes(args.input_file1, args.input_file2)
    
    plot_hypervolume_development(hypervolume1, args.output_path)