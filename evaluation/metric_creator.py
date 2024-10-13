# This file takes a folder with different .json files 
# which each contain a pareto front found by wandb
# and calculates the metrics for each front and saves them in a .csv file

import os
import json
import numpy as np
import pandas as pd
from stats.performance_indicators import hypervolume, spread, generational_distance, inverted_generational_distance
import argparse

def calculate_metrics(folder_path, output_path):
    # First, we get all the files in the folder
    files = os.listdir(folder_path)
    # Only keep the .json files
    files = [file for file in files if file.endswith(".json")]
    # Open each file and calculate the metrics
    metrics = []
    for file in files:
        with open(os.path.join(folder_path, file), "r") as f:
            data = json.load(f)
            # Get the pareto front
            pareto_front = np.array(data["data"])
            # Transform the array into a set of unique points
            pareto_front = np.unique(pareto_front, axis=0)
            # Calculate the metrics
            hv = hypervolume(pareto_front)
            sp = spread(pareto_front)
            igd = inverted_generational_distance(pareto_front)
            # Save the metrics
            metrics.append([pareto_front, hv, sp, gd, igd])

def main():
    # Take the inputs as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", help="Path to the folder containing the .json files", type=str)
    parser.add_argument("--output_path", help="Path to the output .csv file", type=str)
    args = parser.parse_args()

    calculate_metrics(args.folder_path, args.output_path)

if __name__ == "__main__":
    main()