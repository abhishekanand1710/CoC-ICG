import re
import json
import argparse
from collections import defaultdict
import os

def parse_git_patch(patch_content):
    files = {}
    file_sections = re.split(r'diff --git ', patch_content)
    if len(file_sections) > 1:
        file_sections = file_sections[1:]
    
    for section in file_sections:
        if not section.strip():
            continue
            
        file_match = re.match(r'a/(.*) b/', section)
        if not file_match:
            continue
            
        filename = file_match.group(1)
        files[filename] = {
            'functions': set(),
            'modified_lines': set()
        }
        
        hunk_matches = list(re.finditer(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)', section))
        
        for i, hunk in enumerate(hunk_matches):
            # Get function name from hunk header if available
            func_name = hunk.group(5).strip()
            if func_name:
                # print(func_name)
                files[filename]['functions'].add(func_name.strip())
            
            old_start = int(hunk.group(1))
            new_start = int(hunk.group(3))
            
            hunk_start = hunk.end() + 1
            hunk_end = len(section)
            if i < len(hunk_matches) - 1:
                hunk_end = hunk_matches[i+1].start()
            
            hunk_content = section[hunk_start:hunk_end]
            
            # Find Python function definitions in the hunk content
            extract_modified_functions(hunk_content, files[filename]['functions'])
            
            old_line = old_start
            new_line = new_start
            
            for line in hunk_content.split('\n'):
                if not line:
                    continue
                
                if line.startswith('+'):
                    files[filename]['modified_lines'].add(f"{filename}:+{new_line}")
                    new_line += 1
                elif line.startswith('-'):
                    files[filename]['modified_lines'].add(f"{filename}:-{old_line}")
                    old_line += 1
                else:
                    old_line += 1
                    new_line += 1
    
    return files

def extract_modified_functions(hunk_content, functions_set):
    # Adjusted regex to match lines that start with a space, '+' or '-' followed by a valid function definition.
    func_pattern = re.compile(r'^[ +-][ \t]*(def|async def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    
    current_signature = None
    for line in hunk_content.splitlines():
        match = func_pattern.match(line)
        if match:
            # Remove the diff marker or context indicator (the first character) and take the rest of the line as the complete signature.
            # Using rstrip() to remove trailing newline and any extra spaces.
            current_signature = line[1:].strip() if line[0] in "+- " else line.rstrip()
            
            # If the function definition itself is modified, record the signature immediately.
            if line[0] in "+-":
                functions_set.add(current_signature)
        else:
            # For modified lines within a function's body, add the current function signature if it exists.
            if line.startswith('+') or line.startswith('-'):
                if current_signature:
                    functions_set.add(current_signature)

def extract_old_changed_lines_by_file(patch):
    diff_file_pattern = re.compile(r'^diff --git a/(.+?) b/')
    hunk_header_pattern = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@')
    
    file_changed_lines = {}
    current_file = None
    current_old_line = None
    in_hunk = False
    count = 0
    for line in patch.splitlines():
        file_match = diff_file_pattern.match(line)
        if file_match:
            # Normalize file name by using what's captured (e.g. "sphinx/builders/manpage.py")
            current_file = file_match.group(1)
            if current_file not in file_changed_lines:
                file_changed_lines[current_file] = set()
            # Reset hunk state when starting a new file block.
            in_hunk = False
            current_old_line = None
            count += 1
            continue
        
        # --- Hunk header detection ---
        hunk_match = hunk_header_pattern.match(line)
        if hunk_match:
            in_hunk = True
            try:
                current_old_line = int(hunk_match.group(1))
            except ValueError:
                current_old_line = None
                print('could not parse old line number from hunk header')
            continue

        if in_hunk and current_file is not None and current_old_line is not None:
            if line.startswith(' '):
                current_old_line += 1
            elif line.startswith('-'):
                file_changed_lines[current_file].add(current_old_line)
                current_old_line += 1
    
    if count == 0 and patch != "":
        print(patch)
        return {}
    
    return file_changed_lines

def are_all_edit_locations_covered(candidate_patch, gold_patch):
    candidate_files = extract_old_changed_lines_by_file(candidate_patch)
    gold_files = extract_old_changed_lines_by_file(gold_patch)

    for file, gold_changed_lines in gold_files.items():
        if file not in candidate_files:
            return False
        
        candidate_changed_lines = candidate_files[file]
        if not gold_changed_lines.issubset(candidate_changed_lines):
            return False
    return True

def check_function_coverage(gold_files, test_files):
    total_functions = 0
    covered_functions = 0
    missing_functions = {}
    
    for file_path, file_data in test_files.items():
        test_functions = file_data['functions']
        total_functions += len(test_functions)
        
        # A file must exist in gold to have any coverage
        if file_path not in gold_files:
            if test_functions:
                missing_functions[file_path] = list(test_functions)
            continue
            
        gold_functions = gold_files[file_path]['functions']
        
        # Find functions in test that are covered by gold
        covered_funcs = gold_functions.intersection(test_functions)
        covered_functions += len(covered_funcs)
        
        # Track missing functions
        missing_funcs = gold_functions - test_functions
        if missing_funcs:
            missing_functions[file_path] = list(missing_funcs)
            return False
    return True

def check_patch_coverage(gold_patch, test_patch):
    gold_files = parse_git_patch(gold_patch)
    test_files = parse_git_patch(test_patch)
    
    # Check if all lines are covered
    # line_coverage = are_all_edit_locations_covered(gold_patch, test_patch)
    
    # Calculate file coverage
    total_files = len(test_files)
    files_covered = all(file in test_files for file in gold_files)
    
    # Calculate function coverage
    functions_covered = check_function_coverage(gold_files, test_files)
    
    # Prepare result
    result = {
        # 'complete_coverage': line_coverage,
        'missing_files': [file for file in test_files if file not in gold_files],
        'functions_covered': functions_covered,
        'files_covered': files_covered
    }
    
    return result

def is_patch_covered(test_patch, gold_patch):
    coverage_info = check_patch_coverage(gold_patch, test_patch)
    return coverage_info

def read_jsonl(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main(args):
    pred_file = args.pred
    gold_file = "swe_bench_cache/dataset.json"
    if args.dataset_split == "verified":
        gold_file = "swe_bench_verified_cache/dataset.json"
    
    if pred_file.endswith('.jsonl'):
        pred_data = read_jsonl(pred_file)
    else:
        with open(pred_file, 'r') as f:
            pred_data = json.loads(f.read())

    with open(gold_file, 'r') as f:
        gold_data = json.loads(f.read())

    model = pred_data[0]["model_name_or_path"]
    run_id = pred_file.split("/")[-2]
    
    pred_patch_dict = {sample["instance_id"]: sample["model_patch"] for sample in pred_data}
    gold_patch_dict = {key: value["patch"] for key, value in gold_data.items()}

    results = {}
    total_file_coverage = 0
    total_function_coverage = 0
    total_line_coverage = 0

    error_count = 0
    for instance_id, pred_patch in pred_patch_dict.items():
        try:
            gold_patch = gold_patch_dict[instance_id]
            coverage_info = is_patch_covered(pred_patch, gold_patch)

            total_file_coverage += coverage_info["files_covered"]
            total_function_coverage += coverage_info["functions_covered"]
            total_line_coverage += are_all_edit_locations_covered(pred_patch, gold_patch)
        except:
            error_count += 1

    print(f"Error count: {error_count}")
    average_file_coverage = total_file_coverage / len(pred_patch_dict)
    average_function_coverage = total_function_coverage / len(pred_patch_dict)
    average_line_coverage = total_line_coverage / len(pred_patch_dict)
    results["average_coverage"] = {
        "file_coverage": average_file_coverage,
        "function_coverage": average_function_coverage,
        "line_coverage": average_line_coverage
    }

    os.makedirs(f"coverage_results/{args.dataset_split}", exist_ok=True)

    with open(f"coverage_results/{run_id}_{model}_coverage.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Coverage results saved to coverage_results/{run_id}_{model}_coverage.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct Loc % Calculator")
    parser.add_argument("-p", "--pred", type=str, help="Path to prediction file", required=True)
    parser.add_argument("-d", "--dataset_split", type=str, help="Dataset split", default="lite")

    args = parser.parse_args()
    main(args)