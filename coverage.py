import re
import json
import argparse
from collections import defaultdict
import os

def parse_git_patch(patch_content):
    files = []
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
        files.append(filename)
    
    return files

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
            current_file = file_match.group(1)
            if current_file not in file_changed_lines:
                file_changed_lines[current_file] = set()
            in_hunk = False
            current_old_line = None
            count += 1
            continue
        
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

def get_line_coverage_for_edits(candidate_patch, gold_patch):
    candidate_files = extract_old_changed_lines_by_file(candidate_patch)
    gold_files = extract_old_changed_lines_by_file(gold_patch)

    for file, gold_changed_lines in gold_files.items():
        if file not in candidate_files:
            return False
        
        candidate_changed_lines = candidate_files[file]
        if not gold_changed_lines.issubset(candidate_changed_lines):
            return False
    return True

def get_file_coverage(test_patch, gold_patch):
    gold_files = parse_git_patch(gold_patch)
    test_files = parse_git_patch(test_patch)
    
    files_covered = all(file in test_files for file in gold_files)
    return files_covered

def read_jsonl(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main(args):
    pred_file = args.pred
    dataset_dir = args.dataset_dir
    gold_file = f"{dataset_dir}/dataset.json"
    
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
    total_line_coverage = 0

    error_count = 0
    for instance_id, pred_patch in pred_patch_dict.items():
        try:
            gold_patch = gold_patch_dict[instance_id]

            total_file_coverage += get_file_coverage(pred_patch, gold_patch)
            total_line_coverage += get_line_coverage_for_edits(pred_patch, gold_patch)
        except:
            error_count += 1

    print(f"Patch parsing error count: {error_count}")
    average_file_coverage = total_file_coverage / len(pred_patch_dict)
    average_line_coverage = total_line_coverage / len(pred_patch_dict)
    results["average_coverage"] = {
        "file_coverage": average_file_coverage,
        "line_coverage": average_line_coverage
    }

    os.makedirs(f"coverage_results", exist_ok=True)
    output_path = f"coverage_results/{run_id}_{model}_coverage.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Coverage results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct Loc % Calculator")
    parser.add_argument("-p", "--pred", type=str, help="Path to prediction file", required=True)
    parser.add_argument("-d", "--dataset_dir", type=str, help="Dataset directory containing gold file", default="swe_bench_lite_cache")

    args = parser.parse_args()
    main(args)