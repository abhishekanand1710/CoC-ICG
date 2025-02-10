import os
import json
from datasets import load_dataset
import git
import json

from pathlib import Path

DATASET_DIR = Path('./swe_bench_cache')
DATASET_SAVE_FILE = Path(DATASET_DIR, 'dataset.json')
REPOS_DIR = Path(DATASET_DIR, 'repos')

def save_dataset(dataset):
    with open(DATASET_SAVE_FILE.resolve(), 'w') as f:
        json.dump(dataset, f)

def load_swe_bench_dataset(dataset: str = 'princeton-nlp/SWE-bench_Lite'):
    if Path(DATASET_DIR, 'dataset.json').exists():
        with open(DATASET_SAVE_FILE.resolve(), 'r') as f:
            return json.load(f)
        
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
    
    save_dataset(instances)
    
    return instances

def checkout_git_repo_at_commit(repo_name, commit):
    repo_dir = f"{repo_name}.git"

    instance_repo = Path(REPOS_DIR, repo_name)

    if not instance_repo.exists():
        repo = git.Repo.clone_from(f"https://github.com/{repo_name}.git", instance_repo.resolve())
    else:
        repo = git.Repo(instance_repo.resolve())

    repo.git.checkout(commit)
    repo.close()
    
    return repo_dir

def prepare_one_instance(instance):
    instance_id = instance["instance_id"]
    base_commit = instance["base_commit"]
    repo_id = instance["repo"]
    problem_statement = instance["problem_statement"]
    
    patch = instance["patch"]

    REPOS_DIR.mkdir(exist_ok=True)

    repo_dir = checkout_git_repo_at_commit(repo_id, base_commit)
    

def process_instances(instances):
    for instance_id, instance in instances.items():
        prepare_one_instance(instance)

def main():
    os.makedirs('./swe_bench_cache', exist_ok=True)
    dataset = load_swe_bench_dataset()
    process_instances(dataset)

if __name__ == '__main__':
    main()