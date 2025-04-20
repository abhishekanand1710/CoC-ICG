import os
import json
from datasets import load_dataset
import git
import json
from tqdm import tqdm
import argparse
from pathlib import Path
from utils.repo_utils import *

def load_swe_bench_dataset(dataset_split, dataset_save_path):
    '''
    Load dataset either from local storage path or from huggingface and persist it to local storage
    '''
    if dataset_save_path.exists():
        print(f"Dataset instances already saved at {dataset_save_path}.")
        with open(dataset_save_path.resolve(), 'r') as f:
            return json.load(f)
        
    dataset = 'princeton-nlp/SWE-bench_Lite'
    if dataset_split == 'verified':
        dataset = 'princeton-nlp/SWE-bench_Verified'
    data = load_dataset(dataset)['test']
    
    instances = {}
    for instance in data:
        instance = {
            'repo': instance['repo'],
            'instance_id': instance['instance_id'],
            'base_commit': instance['base_commit'],
            'patch': instance['patch'],
            'test_patch': instance['test_patch'],
            'hints_text': instance['hints_text'],
            'problem_statement': instance['problem_statement'],
            'version': instance['version'],
            'base_commit': instance['base_commit'],
            'environment_setup_commit': instance['environment_setup_commit'],
            'created_at': instance['created_at'],
            'FAIL_TO_PASS': json.loads(instance['FAIL_TO_PASS']),
            'PASS_TO_PASS': json.loads(instance['PASS_TO_PASS'])
        }
        instances[instance['instance_id']] = instance

    with open(dataset_save_path.resolve(), 'w') as f:
        json.dump(instances, f)
    print(f"Fetched instances and saved to {dataset_save_path}.")
    
    return instances

def main(args):
    '''
    Load a dataset split and clone its repos to local storage
    '''
    dataset_split = args.split
    dataset_dir = args.dataset_dir
    dataset_save_path = Path(dataset_dir, 'dataset.json')
    repos_dir = Path(dataset_dir, 'repos')

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(repos_dir, exist_ok=True)
    dataset_instances = load_swe_bench_dataset(dataset_split, dataset_save_path)

    print(f"Cloning repos..")
    for _, instance in tqdm(dataset_instances.items()):
        base_commit = instance["base_commit"]
        repo_id = instance["repo"]
        checkout_git_repo_at_commit(repos_dir, repo_id, base_commit)
    print(f"Dataset saved to path: {dataset_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prefetch dataset repositories")
    parser.add_argument("-d", "--dataset_dir", type=str, help="Directory containing SWE-Bench task repos", default="./swe_bench_verified_cache")
    parser.add_argument("-s", "--split", type=str, help="Dataset split to load - lite/verified", required=True)

    args = parser.parse_args()
    main(args)