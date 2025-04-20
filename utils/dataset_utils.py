import os
import json
import pandas as pd

def load_processed_instances(output_file):
    '''
    Loads processed instances for a run
    '''
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            processed_instances = {entry["instance_id"]: entry for entry in json.load(f)}
        return processed_instances
    return {}

def load_local_dataset(input_file, processed_instances):
    '''
    Loads dataset and filters out processed instances
    '''
    with open(input_file, "r") as f:
        dataset = json.load(f)
    
    dataset = {key: value for key, value in dataset.items() if key not in processed_instances}
    return dataset