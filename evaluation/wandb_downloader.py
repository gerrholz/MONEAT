# This file downloads all found pareto fronts from a 
# specific wandb project and saves them in a folder

import wandb
import os
import argparse
import re
import shutil

def extract_number_from_filename(file_path):
    # Extract just the filename, not the full path
    filename = os.path.basename(file_path)
    match = re.search(r'front_(\d+)_', filename)
    return int(match.group(1))

def download_pareto_fronts(project, entity, folder_path):
    # Login to wandb
    api_key = os.getenv("WANDB_API")
    wandb.login(key=api_key)
    # Get the project
    api = wandb.Api()

    runs = api.runs(f"{entity}/{project}")
    print(runs)
    # Get all runs
    #runs = project.runs()
    os.makedirs(folder_path, exist_ok=True)
    # For each run, get the pareto front and save it
    for run in runs:
        # Filter by run name
        if "mo-halfcheetah-v4__CAPQL" not in run.name:
            continue
        files = [file.name for file in run.files(per_page=200) if "media/table/eval/front_" in file.name]
        #print(files)
        if files:
            # Sort files to get the latest one, assuming they are sequentially named
            files.sort(key=extract_number_from_filename)
            latest_file_name = files[-1]

            # Check if the filename already exists in the folder
            existing_file_path = os.path.join(folder_path, latest_file_name)
            if os.path.exists(existing_file_path):
                # Extract the file name and extension
                file_name, file_extension = os.path.splitext(latest_file_name)
                counter = 1

                # Construct a new name for the existing file
                new_existing_file_name = f"{file_name}_old_{counter}{file_extension}"
                new_existing_file_path = os.path.join(folder_path, new_existing_file_name)

                # Ensure the new name is unique
                while os.path.exists(new_existing_file_path):
                    counter += 1
                    new_existing_file_name = f"{file_name}_old_{counter}{file_extension}"
                    new_existing_file_path = os.path.join(folder_path, new_existing_file_name)

                # Rename the existing file
                shutil.move(existing_file_path, new_existing_file_path)

            # Download the latest Pareto front file
            latest_file = run.file(latest_file_name)
            latest_file.download(replace=True, root=folder_path)

            print(f"Downloaded {latest_file_name} from run {run.id}")

def flatten_directory_structure(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            source_path = os.path.join(dirpath, filename)
            target_path = os.path.join(root_dir, filename)
            if source_path != target_path:
                shutil.move(source_path, target_path)

    # Optionally remove subdirectories if they are empty
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)

if __name__ == "__main__":
    # Take the inputs as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Name of the wandb project", type=str, required=True)
    parser.add_argument("--folder_path", help="Path to the folder where the pareto fronts will be saved", type=str, required=True)
    parser.add_argument("--entity", help="Name of the entity", type=str, required=True)

    args = parser.parse_args()
    download_pareto_fronts(args.project, args.entity, args.folder_path)
    flatten_directory_structure(args.folder_path)
    
