import re
from collections import defaultdict
import argparse
import json

import re

def parse_patch_extended(patch):
    """
    Parse a unified diff (git patch) and return a dictionary mapping file names to a list of hunks.
    Each hunk is represented as a dictionary with:
      - 'header': The original hunk header line.
      - 'function': The function context (if provided, typically trailing the hunk header).
      - 'new_start': The starting line number in the new file.
      - 'new_length': The number of lines in the new file hunk.
      - 'new_end': The last line number in the new file hunk.
      - 'lines': The list of changed lines (actual diff lines starting with '+' or '-' but not file markers).
    """
    files = {}
    current_file = None
    # Regex for hunk header to capture old start & length, new start & length, and trailing function context.
    hunk_header_pattern = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$')
    lines = patch.splitlines()
    
    for line in lines:
        # Detect a new file being modified.
        if line.startswith('diff --git'):
            tokens = line.split()
            if len(tokens) >= 4:
                # Convention: tokens[3] is "b/<filename>"; remove the "b/" prefix.
                current_file = tokens[3][2:]
                files[current_file] = []
        # Detect a hunk header.
        elif current_file is not None and line.startswith('@@'):
            m = hunk_header_pattern.match(line)
            if m:
                # Extract numbers from the header.
                new_start = int(m.group(3))
                new_length = int(m.group(4)) if m.group(4) is not None else 1
                function_context = m.group(5).strip()
                hunk = {
                    'header': line,
                    'function': function_context,
                    'new_start': new_start,
                    'new_length': new_length,
                    'new_end': new_start + new_length - 1,
                    'lines': []  # Optionally store the changed lines.
                }
                files[current_file].append(hunk)
        # Optionally, capture the actual changed lines (ignoring file markers like '+++' and '---')
        elif current_file is not None and files[current_file] and (line.startswith('+') or line.startswith('-')):
            if not (line.startswith('+++') or line.startswith('---')):
                files[current_file][-1]['lines'].append(line)
    return files

def patch_coverage_details(candidate_patch, gold_patch):
    """
    Compare a candidate patch against a gold patch and return a tuple of three booleans:
      (files_covered, functions_covered, line_numbers_covered)

    - files_covered is True if every file modified in gold_patch is also modified in candidate_patch.
    
    - functions_covered is True if for every hunk in gold_patch that has a non-empty function context,
      there is at least one hunk in the corresponding file in candidate_patch with the same function context.
    
    - line_numbers_covered is True if for every hunk in gold_patch there is at least one hunk in candidate_patch
      that fully covers its new file line number range (i.e. the candidate's new_start is less than or equal to
      the gold's new_start, and the candidate's new_end is greater than or equal to the gold's new_end).
    """
    candidate_files = parse_patch_extended(candidate_patch)
    gold_files = parse_patch_extended(gold_patch)
    
    files_covered = True
    functions_covered = True
    line_numbers_covered = True
    
    # Iterate over every file present in the gold patch.
    for file, gold_hunks in gold_files.items():
        if file not in candidate_files:
            files_covered = False
            # If the file is missing, then none of the hunks (thus functions and lines) can be covered.
            functions_covered = False
            line_numbers_covered = False
            continue
        
        candidate_hunks = candidate_files[file]
        for gold_hunk in gold_hunks:
            # Check function coverage if the gold hunk provides a function context.
            if gold_hunk['function']:
                func_match = any(candidate_hunk.get('function', '') == gold_hunk['function']
                                 for candidate_hunk in candidate_hunks)
                if not func_match:
                    functions_covered = False
            
            # Check that there is at least one candidate hunk whose new file line range completely covers the gold hunk.
            line_match = any(candidate_hunk['new_start'] <= gold_hunk['new_start'] and
                             candidate_hunk['new_end'] >= gold_hunk['new_end']
                             for candidate_hunk in candidate_hunks)
            if not line_match:
                line_numbers_covered = False
                
    return files_covered, functions_covered, line_numbers_covered


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
        file_covered, function_covered, lines_covered = patch_coverage_details(pred_patch, gold_patch)
        # results["per_instance_coverage"][instance_id] = {
        #     "is_covered": is_covered,
        #     "coverage_info": coverage_info
        # }
        total_file_coverage += file_covered
        total_function_coverage += function_covered
        total_line_coverage += lines_covered

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