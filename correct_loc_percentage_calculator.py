import re
from collections import defaultdict
import argparse
import json

import re
from collections import defaultdict

import re

import re

def parse_git_patch(patch_content):
    """Parse git patch and extract modified files, functions and lines."""
    files = {}
    
    # Split patch into files
    file_sections = re.split(r'diff --git ', patch_content)
    if len(file_sections) > 1:
        file_sections = file_sections[1:]  # Skip empty first section
    
    for section in file_sections:
        if not section.strip():
            continue
            
        # Extract filename
        file_match = re.match(r'a/(.*) b/', section)
        if not file_match:
            continue
            
        filename = file_match.group(1)
        files[filename] = {
            'functions': set(),
            'lines': set()
        }
        
        # Get function names and lines from hunks
        hunk_matches = list(re.finditer(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)', section))
        
        for i, hunk in enumerate(hunk_matches):
            func_name = hunk.group(5).strip()
            if func_name:
                files[filename]['functions'].add(func_name)
            
            # Get hunk content
            hunk_start = hunk.end() + 1
            hunk_end = len(section)
            if i < len(hunk_matches) - 1:
                hunk_end = hunk_matches[i+1].start()
            
            hunk_content = section[hunk_start:hunk_end]
            
            # Extract lines from hunk
            for line in hunk_content.split('\n'):
                if line.startswith('+') or line.startswith('-'):
                    # Just store the line type and content, no line numbers
                    files[filename]['lines'].add((line[0], line[1:].strip()))
    
    return files

def check_patch_coverage(patch1, patch2):
    """Check if patch1 covers patch2, returning detailed information."""
    patch1_files = parse_git_patch(patch1)
    patch2_files = parse_git_patch(patch2)
    
    result = {
        'complete_coverage': True,
        'missing_files': [],
        'missing_functions': {},
        'missing_lines': {},
        'coverage': {
            'files': {'total': len(patch2_files), 'covered': 0},
            'functions': {'total': 0, 'covered': 0},
            'lines': {'total': 0, 'covered': 0}
        }
    }
    
    # Count total functions and lines in patch2
    for file_path, file_data in patch2_files.items():
        result['coverage']['functions']['total'] += len(file_data['functions'])
        result['coverage']['lines']['total'] += len(file_data['lines'])
    
    # Check coverage by file
    for file_path, file_data in patch2_files.items():
        if file_path not in patch1_files:
            result['complete_coverage'] = False
            result['missing_files'].append(file_path)
            continue
        
        result['coverage']['files']['covered'] += 1
        
        # Check function coverage
        for func_name in file_data['functions']:
            if func_name in patch1_files[file_path]['functions']:
                result['coverage']['functions']['covered'] += 1
            else:
                result['complete_coverage'] = False
                if file_path not in result['missing_functions']:
                    result['missing_functions'][file_path] = []
                result['missing_functions'][file_path].append(func_name)
        
        # Check line coverage
        file_covered_lines = 0
        for line in file_data['lines']:
            if line in patch1_files[file_path]['lines']:
                file_covered_lines += 1
            else:
                result['complete_coverage'] = False
                if file_path not in result['missing_lines']:
                    result['missing_lines'][file_path] = []
                result['missing_lines'][file_path].append(line)
        
        result['coverage']['lines']['covered'] += file_covered_lines
    
    # Calculate coverage percentages
    for metric in ['files', 'functions', 'lines']:
        total = result['coverage'][metric]['total']
        covered = result['coverage'][metric]['covered']
        result['coverage'][metric]['percentage'] = (covered / total * 100) if total > 0 else 0
    
    return result

def is_patch_covered(covering_patch, covered_patch):
    """
    Check if covering_patch covers all files, functions, and lines in covered_patch.
    Returns a tuple of (boolean, coverage_data).
    """
    coverage_info = check_patch_coverage(covering_patch, covered_patch)
    return coverage_info['complete_coverage'], coverage_info

def read_jsonl(file_path):
        data = []
        with open(file_path) as f:
            for line in f:
                data.append(json.loads(line))
        return data

def main(args):
    pred_file = args.pred
    output_file = args.gold
    
    if pred_file.endswith('.jsonl'):
        pred_data = read_jsonl(pred_file)
    else:
        with open(pred_file, 'r') as f:
            pred_data = json.loads(f.read())


    with open(output_file, 'r') as f:
        gold_data = json.loads(f.read())

    model = pred_data[0]["model_name_or_path"]
    run_id = pred_file.split("/")[-2]
    
    pred_patch_dict = {sample["instance_id"]: sample["model_patch"] for sample in pred_data}
    gold_patch_dict = {key: value["patch"] for key, value in gold_data.items()}

    results = {"per_instance_coverage": {}}
    total_file_coverage = 0
    total_function_coverage = 0
    total_line_coverage = 0

    for instance_id, pred_patch in pred_patch_dict.items():
        gold_patch = gold_patch_dict[instance_id]
        is_covered, coverage_info = is_patch_covered(pred_patch, gold_patch)
        results["per_instance_coverage"][instance_id] = {
            "is_covered": is_covered,
            "coverage_info": coverage_info
        }
        total_file_coverage += coverage_info["coverage"]["files"]["percentage"]
        total_function_coverage += coverage_info["coverage"]["functions"]["percentage"]
        total_line_coverage += coverage_info["coverage"]["lines"]["percentage"]

    average_file_coverage = total_file_coverage / len(pred_patch_dict)
    average_function_coverage = total_function_coverage / len(pred_patch_dict)
    average_line_coverage = total_line_coverage / len(pred_patch_dict)
    results["average_coverage"] = {
        "file_coverage": average_file_coverage,
        "function_coverage": average_function_coverage,
        "line_coverage": average_line_coverage
    }

    with open(f"coverage_results/{run_id}_{model}_coverage.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Coverage results saved to coverage_results/{run_id}_{model}_coverage.json")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct Loc % Calculator")
    parser.add_argument("-p", "--pred", type=str, help="Path to prediction file", required=True)
    parser.add_argument("-g", "--gold", type=str, help="Path to gold file", required=True)

    args = parser.parse_args()
    main(args)