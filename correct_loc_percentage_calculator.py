import re
from collections import defaultdict
import argparse
import json

import re
from collections import defaultdict

import re

def parse_git_patch(patch_content):
    """Parse git patch and extract modified files with their line changes."""
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
            'modified_lines': set()  # Will store line identifiers of modified lines
        }
        
        # Get function names and line changes from hunks
        hunk_matches = list(re.finditer(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)', section))
        
        for i, hunk in enumerate(hunk_matches):
            func_name = hunk.group(5).strip()
            if func_name:
                files[filename]['functions'].add(func_name)
            
            # Parse the hunk header
            old_start = int(hunk.group(1))
            old_count = int(hunk.group(2)) if hunk.group(2) else 1
            new_start = int(hunk.group(3))
            new_count = int(hunk.group(4)) if hunk.group(4) else 1
            
            # Get hunk content
            hunk_start = hunk.end() + 1
            hunk_end = len(section)
            if i < len(hunk_matches) - 1:
                hunk_end = hunk_matches[i+1].start()
            
            hunk_content = section[hunk_start:hunk_end]
            
            # Process line changes in the hunk
            old_line = old_start
            new_line = new_start
            
            for line in hunk_content.split('\n'):
                if not line:
                    continue
                
                if line.startswith('+'):
                    # Addition - record the line in the new file
                    files[filename]['modified_lines'].add(f"{filename}:+{new_line}")
                    new_line += 1
                elif line.startswith('-'):
                    # Deletion - record the line in the old file
                    files[filename]['modified_lines'].add(f"{filename}:-{old_line}")
                    old_line += 1
                else:
                    # Context line - increment both counters
                    old_line += 1
                    new_line += 1
    
    return files

def check_patch_coverage(gold_patch, test_patch):
    """
    Check if gold_patch covers test_patch by verifying that all lines
    modified in test_patch are also modified in gold_patch.
    """
    gold_files = parse_git_patch(gold_patch)
    test_files = parse_git_patch(test_patch)
    
    # Collect all modified lines from both patches
    gold_modified_lines = set()
    for file_data in gold_files.values():
        gold_modified_lines.update(file_data['modified_lines'])
    
    test_modified_lines = set()
    for file_data in test_files.values():
        test_modified_lines.update(file_data['modified_lines'])
    
    # Check if all test patch lines are covered by gold patch
    missing_lines = test_modified_lines - gold_modified_lines
    
    # Organize results by file
    result = {
        'complete_coverage': len(missing_lines) == 0,
        'missing_files': [],
        'missing_functions': {},
        'missing_lines': {},
        'coverage_summary': {
            'files': {
                'total': len(test_files),
                'covered': sum(1 for file in test_files if file in gold_files)
            },
            'lines': {
                'total': len(test_modified_lines),
                'covered': len(test_modified_lines - missing_lines)
            }
        }
    }
    
    # Identify missing files
    result['missing_files'] = [file for file in test_files if file not in gold_files]
    
    # Organize missing lines by file
    for line_id in missing_lines:
        file_path, line_info = line_id.split(':')
        if file_path not in result['missing_lines']:
            result['missing_lines'][file_path] = []
        result['missing_lines'][file_path].append(line_info)
    
    # Check function coverage
    total_functions = 0
    covered_functions = 0
    
    for file_path, file_data in test_files.items():
        file_functions = file_data['functions']
        total_functions += len(file_functions)
        
        if file_path in gold_files:
            gold_functions = gold_files[file_path]['functions']
            missing_functions = file_functions - gold_functions
            
            if missing_functions:
                result['missing_functions'][file_path] = list(missing_functions)
            
            covered_functions += len(file_functions) - len(missing_functions)
    
    # Add function coverage to summary
    result['coverage_summary']['functions'] = {
        'total': total_functions,
        'covered': covered_functions
    }
    
    # Calculate coverage percentages
    for metric in ['files', 'functions', 'lines']:
        total = result['coverage_summary'][metric]['total']
        covered = result['coverage_summary'][metric]['covered']
        result['coverage_summary'][metric]['percentage'] = (covered / total * 100) if total > 0 else 100
    
    return result

def is_patch_covered(gold_patch, test_patch):
    """
    Check if gold_patch covers all line edits in test_patch.
    Returns a tuple of (boolean, coverage_data).
    """
    coverage_info = check_patch_coverage(gold_patch, test_patch)
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
        total_file_coverage += coverage_info["coverage_summary"]["files"]["percentage"]
        total_function_coverage += coverage_info["coverage_summary"]["functions"]["percentage"]
        total_line_coverage += coverage_info["coverage_summary"]["lines"]["percentage"]

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