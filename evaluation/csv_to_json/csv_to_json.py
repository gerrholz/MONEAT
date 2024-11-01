import os
import pandas as pd
import json

def convert_csv_to_json(input_folder, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each CSV file in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            csv_path = os.path.join(input_folder, filename)
            json_filename = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join(output_folder, json_filename)

            # Read the CSV file
            df = pd.read_csv(csv_path)

            # Create the JSON structure
            json_data = {
                "columns": df.columns.tolist(),
                "data": df.values.tolist()
            }

            # Save the JSON data to a file
            with open(json_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)

            print(f"Converted {csv_path} to {json_path}")

# Specify your input and output directories
input_folder = 'csvs'
output_folder = '../fronts/pgmorl_cheetah'

convert_csv_to_json(input_folder, output_folder)