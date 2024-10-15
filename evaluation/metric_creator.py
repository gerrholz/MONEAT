# This file takes a folder with different .json files 
# which each contain a pareto front found by wandb
# and calculates the metrics for each front and saves them in a .json file

import os
import json
import numpy as np
import pandas as pd
from performance_indicators import hypervolume, spacing, sparsity, inverted_generational_distance
import argparse

def calculate_metrics(name, folder_path, output_path, has_known_pareto_front=True):
    # First, we get all the files in the folder
    files = os.listdir(folder_path)
    # Only keep the .json files
    files = [file for file in files if file.endswith(".json")]
    # Open each file and calculate the metrics
    hypervolumes = []
    cardinalities = []
    idgs = []
    spacings = []
    sparsities = []
    fronts = []
    for file in files:
        with open(os.path.join(folder_path, file), "r") as f:
            data = json.load(f)
            # Get the pareto front
            pareto_front = np.array(data["data"])

    
            # Transform the array into a set of unique points
            pareto_front = np.unique(pareto_front, axis=0)
            # Calculate the metrics
            hv = hypervolume(np.array([-100,-100]),pareto_front)
            s = sparsity(pareto_front)
            sp = spacing(pareto_front)
            cardinality = len(pareto_front)

            # If we have a known pareto front, calculate the generational distance and inverted generational distance
            if has_known_pareto_front:
                # Load front from file
                with open("fronts/swimmer_front.json", "r") as f:
                    known_pareto_front = np.array(json.load(f)["data"])
                    idg = inverted_generational_distance(pareto_front, known_pareto_front)
            else: 
                idg = None
            # Save the metrics
            hypervolumes.append(hv)
            cardinalities.append(cardinality)
            spacings.append(s)
            sparsities.append(sp)
            idgs.append(idg)
            fronts.append(pareto_front.tolist())

    # Save the metrics to a json file
    metrics = {
        "name": name,
        "hypervolume": hypervolumes,
        "cardinality": cardinalities,
        "spacing": spacings,
        "sparsity": sparsities,
        "inverted_generational_distance": idgs,
        "pareto_fronts": fronts
    }

    with open(output_path, "w") as f:
        json.dump(metrics, f)

def main():
    # Take the inputs as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the experiment", type=str, required=True)
    parser.add_argument("--folder_path", help="Path to the folder containing the .json files", type=str)
    parser.add_argument("--output_path", help="Path to the output .json file", type=str)
    args = parser.parse_args()

    calculate_metrics(args.name, args.folder_path, args.output_path, True)

if __name__ == "__main__":
    main()