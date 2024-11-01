"""
This file takes two metrics files and plots the metrics against each other
and saves the plots in a folder
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

from scipy.interpolate import interp1d

import seaborn as sns

def calulate_mean_std_metric(metric):
    # Calculate the mean and std of the metric
    metrics = np.array(metric)
    mean_metric = np.mean(metrics)
    std_metric = np.std(metrics)
    return mean_metric, std_metric

def calulate_mean_std_fronts(pareto_fronts):
    # Determine the number of points for interpolation
    num_points = max(len(front) for front in pareto_fronts)
    
    # Create arrays to store interpolated fronts
    interpolated_fronts = []

    for front in pareto_fronts:
        # Extract x and y coordinates
        y, x  = zip(*front)
        
        # Create interpolation functions
        interp_func_x = interp1d(np.arange(len(x)), x, kind='linear', fill_value='extrapolate')
        interp_func_y = interp1d(np.arange(len(y)), y, kind='linear', fill_value='extrapolate')
        
        # Generate interpolated values
        new_x = interp_func_x(np.linspace(0, len(x) - 1, num_points))
        new_y = interp_func_y(np.linspace(0, len(y) - 1, num_points))
        
        # Append the interpolated front
        interpolated_fronts.append(np.column_stack((new_x, new_y)))
    
    # Convert to array
    stacked_fronts = np.array(interpolated_fronts)
    
    # Calculate mean and std
    mean_front = np.nanmean(stacked_fronts, axis=0)
    std_front = np.nanstd(stacked_fronts, axis=0)
    
    return mean_front, std_front

def plot_pareto_front(mean_front, std_front, name):
    x = mean_front[:, 0]
    y = mean_front[:, 1]

    std_x = std_front[:, 0]
    std_y = std_front[:, 1]

    plt.plot(x, y, label=name, linewidth=2, linestyle='--', marker='o', markersize=5, markerfacecolor='white')
    plt.fill_between(x, y - std_y, y + std_y, alpha=0.2)



def plot_errorbar(mean_metric1, std_metric1, mean_metric2, std_metric2, mean_metric3, std_metric3, mean_metric4, std_metric4, name1, name2, name3, name4, metric_name):
    # Define positions for the error bars
    sns.set_theme(style="whitegrid", palette="pastel")
    positions = np.array([1, 2, 3, 4])
    
    # Choose aesthetically pleasing colors
    # Colors for each algorithm
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plt.rcParams['figure.dpi'] = 360
    
    # Plot error bars with enhanced visuals
    plt.errorbar(positions[0], mean_metric1, yerr=std_metric1, 
                 fmt='o', ecolor=colors[0], elinewidth=2, capsize=5, capthick=2, markersize=8, markerfacecolor='white', markeredgewidth=2, color=colors[0], label=name1)

    plt.errorbar(positions[1], mean_metric2, yerr=std_metric2, 
                 fmt='o', ecolor=colors[1], elinewidth=2, capsize=5, capthick=2, markersize=8, markerfacecolor='white', markeredgewidth=2, color=colors[1], label=name2)
    
    print(mean_metric3, std_metric3)
    plt.errorbar(positions[2], mean_metric3, yerr=std_metric3,
                    fmt='o', ecolor=colors[2], elinewidth=2, capsize=5, capthick=2, markersize=8, markerfacecolor='white', markeredgewidth=2, color=colors[2], label=name3)
    
    plt.errorbar(positions[3], mean_metric4, yerr=std_metric4,
                    fmt='o', ecolor=colors[3], elinewidth=2, capsize=5, capthick=2, markersize=8, markerfacecolor='white', markeredgewidth=2, color=colors[3],
                    label=name4)
    
    
    # Add labels and title
    plt.xticks(positions, [name1, name2, name3, name4], fontsize=10)
    plt.ylabel(metric_name, fontsize=10)
    plt.title(f"Mean {metric_name} for {name1}, {name2}, {name3}, and {name4}", fontsize=12, fontweight='bold')
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend
    plt.legend([name1, name2, name3, name4], loc='best', fontsize=10)
    sns.despine(left=True)

    
    # Add a background color
    #plt.gca().set_facecolor('#f7f7f7')
    
    # Ensure layout fits well
    plt.tight_layout()

    # Show the plot
    #plt.show()

def plot_metrics(metric_file1, metric_file2, metric_file3, metric_file4, output_folder):
    # Load the metrics
    with open(metric_file1, "r") as f:
        data1 = json.load(f)
    with open(metric_file2, "r") as f:
        data2 = json.load(f)
    with open(metric_file3, "r") as f:
        data3 = json.load(f)
    with open(metric_file4, "r") as f:
        data4 = json.load(f)
    
    # Get the names of the algorithms
    name1 = data1["name"]
    name2 = data2["name"]
    name3 = data3["name"]
    name4 = data4["name"]


    # Plot the pareto fronts with std
    front_1 = data1["pareto_fronts"]
    front_2 = data2["pareto_fronts"]
    front_3 = data3["pareto_fronts"]
    front_4 = data4["pareto_fronts"]

    mean_front_1, std_front_1 = calulate_mean_std_fronts(front_1)
    mean_front_2, std_front_2 = calulate_mean_std_fronts(front_2)
    mean_front_3, std_front_3 = calulate_mean_std_fronts(front_3)
    mean_front_4, std_front_4 = calulate_mean_std_fronts(front_4)

    plot_pareto_front(mean_front_1, std_front_1, name1)
    plot_pareto_front(mean_front_2, std_front_2, name2)
    plot_pareto_front(mean_front_3, std_front_3, name3)
    plot_pareto_front(mean_front_4, std_front_4, name4)
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.rcParams['figure.dpi'] = 360


    plt.xlabel("Control cost", fontsize=10)
    plt.ylabel("Reward for moving forward", fontsize=10)
    plt.title("Average pareto fronts for {}, {}, {}, and {} ".format(name1, name2,name3,name4), fontsize=12, fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.legend(loc='best', fontsize=10)

    #plt.gca().set_facecolor('#f7f7f7')
    sns.despine(left=True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, "pareto_fronts.png"))
    plt.close()

    # Plot the hypervolume
    hypervolumes1 = data1["hypervolume"]
    hypervolumes2 = data2["hypervolume"]
    hypervolumes3 = data3["hypervolume"]
    hypervolumes4 = data4["hypervolume"]

    mean_hv1, std_hv1 = calulate_mean_std_metric(hypervolumes1)
    mean_hv2, std_hv2 = calulate_mean_std_metric(hypervolumes2)
    mean_hv3, std_hv3 = calulate_mean_std_metric(hypervolumes3)
    mean_hv4, std_hv4 = calulate_mean_std_metric(hypervolumes4)

    plot_errorbar(mean_hv1, std_hv1, mean_hv2, std_hv2, mean_hv3, std_hv3, mean_hv4, std_hv4, name1, name2, name3, name4, "Hypervolume")
    plt.savefig(os.path.join(output_folder, "hypervolume.png"))
    plt.close()

    # Plot the cardinality
    cardinalities1 = data1["cardinality"]
    cardinalities2 = data2["cardinality"]

    mean_card1, std_card1 = calulate_mean_std_metric(cardinalities1)
    mean_card2, std_card2 = calulate_mean_std_metric(cardinalities2)

    plot_errorbar(mean_card1, std_card1, mean_card2, std_card2, name1, name2, "Cardinality")
    plt.savefig(os.path.join(output_folder, "cardinality.png"))
    plt.close()

    # Plot the sparsity
    sparsities1 = data1["sparsity"]
    sparsities2 = data2["sparsity"]

    mean_sp1, std_sp1 = calulate_mean_std_metric(sparsities1)
    mean_sp2, std_sp2 = calulate_mean_std_metric(sparsities2)

    plot_errorbar(mean_sp1, std_sp1, mean_sp2, std_sp2, name1, name2, "Sparsity")
    plt.savefig(os.path.join(output_folder, "sparsity.png"))
    plt.close()

    # Plot the spacing
    spacings1 = data1["spacing"]
    spacings2 = data2["spacing"]

    mean_s1, std_s1 = calulate_mean_std_metric(spacings1)
    mean_s2, std_s2 = calulate_mean_std_metric(spacings2)

    plot_errorbar(mean_s1, std_s1, mean_s2, std_s2, name1, name2, "Spacing")
    plt.savefig(os.path.join(output_folder, "spacing.png"))
    plt.close()

    # Plot the inverted generational distance
    idgs1 = data1["inverted_generational_distance"]
    idgs2 = data2["inverted_generational_distance"]

    # Check if idgs is none
    if not None in idgs1 and not None in idgs2:
        mean_idg1, std_idg1 = calulate_mean_std_metric(idgs1)
        mean_idg2, std_idg2 = calulate_mean_std_metric(idgs2)

        print(mean_idg1, std_idg1, mean_idg2, std_idg2)

        plot_errorbar(mean_idg1, std_idg1, mean_idg2, std_idg2, name1, name2, "Inverted Generational Distance")
        plt.savefig(os.path.join(output_folder, "inverted_generational_distance.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_file1", help="Path to the first metric file", type=str, required=True)
    parser.add_argument("--metric_file2", help="Path to the second metric file", type=str, required=True)
    parser.add_argument("--metric_file3", help="Path to the third metric file", type=str, required=True)
    parser.add_argument("--metric_file4", help="Path to the fourth metric file", type=str, required=True)
    parser.add_argument("--output_folder", help="Path to the folder where the plots will be saved", type=str, required=True)
    args = parser.parse_args()
    plot_metrics(args.metric_file1, args.metric_file2, args.metric_file3, args.metric_file4, args.output_folder)

