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

def extract_number_from_filename(file_path):
    filename = os.path.basename(file_path)
    match = re.search(r'front_(\d+)_', filename)
    return int(match.group(1))

def calculate_hypervolume(pareto_front):
    # Assume the reference point is at zero for simplicity
    reference_point = [-100, -100]
    
    return hypervolume(np.array([-100,-100]), pareto_front)

def download_pareto_fronts(project, entity, folder_path):
    api_key = os.getenv("WANDB_API")
    wandb.login(key=api_key)
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    os.makedirs(folder_path, exist_ok=True)

    all_hypervolumes = []

    for run in runs:
        # Filter if run name contains "eval"
        #if "mo-halfcheetah-v4__PGMORL" not in run.name:
        #    continue
        files = [file.name for file in run.files(per_page=800) if "media/table/eval/front_" in file.name]
        if files:
            files.sort(key=extract_number_from_filename)
            hypervolumes = []

            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                run.file(file_name).download(replace=True, root=folder_path)

                with open(file_path, 'r') as f:
                    pareto_front = json.load(f)

                hypervolume = calculate_hypervolume(np.array(pareto_front["data"]))
                hypervolumes.append(hypervolume)
            #hypervolumes = np.sort(hypervolumes)

            all_hypervolumes.append(hypervolumes)
        print(f"Downloaded Pareto fronts for run {run.id}")

    return all_hypervolumes

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

    plt.plot(iterations, mean_hv, label='Mean Hypervolume', color='orange', linewidth=2, linestyle='--', marker='o', markersize=5, markerfacecolor='white')
    plt.fill_between(iterations, mean_hv - std_hv, mean_hv + std_hv, color='orange', alpha=0.2, label='Std Deviation')

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Name of the wandb project", type=str, required=True)
    parser.add_argument("--folder_path", help="Path to the folder where the pareto fronts will be saved", type=str, required=True)
    parser.add_argument("--entity", help="Name of the entity", type=str, required=True)
    parser.add_argument("--output_path", help="Path to file where the hypervolumes will be saved", type=str, required=True)

    args = parser.parse_args()

    hypervolumes = download_pareto_fronts(args.project, args.entity, args.folder_path) 
    # Save hypervolumes in a json file
    with open(args.output_path, "w+") as f:
        json.dump(hypervolumes, f)

    #plot_hypervolume_development(hypervolumes, args.output_folder)