import os
import json
from typing import Dict, List, Optional, Any
from utils.repo_utils import *
from tree_sitter import Language, Parser
import tree_sitter_python
import difflib

PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

def extract_codebase_context(codebase_path, max_files = None):
    '''
    Parses code files from a repository and extracts modules/files, classes, functions, and file structure
    '''
    context = {
        "modules": {},
        "classes": {},
        "functions": {},
        "file_structure": {}
    }
    
    function_query = PY_LANGUAGE.query("""
        (function_definition
          name: (identifier) @function_name
          parameters: (parameters) @params
          body: (block) @body
          [
            (string) @docstring
          ]?
        )
    """)
    
    class_query = PY_LANGUAGE.query("""
        (class_definition
          name: (identifier) @class_name
          body: (block) @body
          [
            (string) @docstring
          ]?
          [
            (argument_list) @parent_classes
          ]?
        )
    """)
    
    python_files = []
    for root, _, files in os.walk(codebase_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    if max_files:
        python_files = python_files[:max_files]
    
    for file_path in python_files:
        if 'tests/' in file_path or file_path.endswith('__init__.py'):
            continue
        rel_path = os.path.relpath(file_path, codebase_path)
        module_name = rel_path.replace('/', '.').replace('\\', '.').replace('.py', '')
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            code = file.read()
            tree = parser.parse(bytes(code, 'utf-8'))
            root_node = tree.root_node
            
            file_info = {
                "path": rel_path,
                "module_name": module_name,
                "functions": [],
                "classes": []
            }

            for match in function_query.matches(root_node):
                function_info = extract_function_info(match, code)
                if not function_info:
                    continue
                function_info["file_path"] = rel_path
                file_info["functions"].append(function_info["name"])
                context["functions"][f"{module_name}.{function_info['name']}"] = function_info

            for match in class_query.matches(root_node):
                class_info = extract_class_info(match, code)
                if not class_info:
                    continue
                class_info["file_path"] = rel_path
                file_info["classes"].append(class_info["name"])
                context["classes"][f"{module_name}.{class_info['name']}"] = class_info
            
            context["file_structure"][rel_path] = file_info
    
    context["modules"] = {val['module_name']: key for key, val in context['file_structure'].items()}

    return context

def extract_function_info(match, code):
    match = match[1:]
    code = code.split('\n')

    function_name_node = next((node['function_name'] for node in match if 'function_name' in node), None)
    params_node = next((node['params'] for node in match if 'params' in node), None) 
    body_node = next((node['body'] for node in match if 'body' in node), None)

    start_row , start_col= function_name_node[0].start_point
    end_row, end_col = function_name_node[0].end_point
    function_name = code[start_row:end_row+1][0][start_col:end_col]
    if function_name in ['__init__', '__repr__', '__call__', '__new__']:
        return None

    params_end_row= params_node[0].end_point.row
    function_sig = code[start_row:params_end_row+1]
    function_sig = '\n'.join(function_sig)
    
    body_end_row= body_node[0].end_point.row
    function_def = code[start_row:body_end_row+1]
    function_def = '\n'.join(function_def)

    return {
        "name": function_name,
        "signature": function_sig,
        "definition": function_def,
        "line_range": (start_row+1, body_end_row+1)
    }

def extract_class_info(match, code):
    code = code.split('\n')
    match = match[1:]

    class_name_node = next((node['class_name'] for node in match if 'class_name' in node), None)
    body_node = next((node['body'] for node in match if 'body' in node), None)

    start_row , start_col= class_name_node[0].start_point
    end_row, end_col = class_name_node[0].end_point

    class_name = code[start_row:end_row+1][0][start_col:end_col]

    body_end_row = body_node[0].end_point.row
    class_def = code[start_row:body_end_row+1]
    class_def = '\n'.join(class_def)
    
    return {
        "name": class_name,
        "definition": class_def,
        "line_range": (start_row+1, body_end_row+1)
    }

def get_file_content(codebase_path, file_path):
    full_path = os.path.join(codebase_path, file_path)
    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
    
def generate_repo_structure(root_path, indent="\t", max_depth=None):
    ignored = ['.git', '__pycache__', '.DS_Store', 'tests']
    
    result = []
    root_name = os.path.basename(os.path.abspath(root_path))
    result.append(f"{root_name}/")
    
    def walk_dir(directory, depth=0):
        if max_depth is not None and depth >= max_depth:
            return
        
        items = []
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if any(p in path for p in ignored):
                continue
            is_dir = os.path.isdir(path)
            is_python = name.endswith('.py')
            
            if is_dir or is_python:
                items.append((name, path, is_dir))
            
        for name, path, is_dir in sorted(items, key=lambda x: (not x[2], x[0].lower())):
            result.append(f"{indent * (depth + 1)}{name}{'/' if is_dir else ''}")
            if is_dir:
                walk_dir(path, depth + 1)
    
    walk_dir(root_path)
    structure = "\n".join(result)
    return structure

def closest_match(target, string_list, is_file=False):
    original_strings = string_list.copy()
    processed_strings = string_list
    if not is_file:
        processed_strings = [s.split('.')[-1] for s in string_list]
    closest = difflib.get_close_matches(target, processed_strings, n=1, cutoff=0)[0]
    match_idx = processed_strings.index(closest)
    return original_strings[match_idx]

def analyze_codebase(codebase_path, output_path='file_structure.json', max_files=None):
    if not os.path.exists(codebase_path):
        raise ValueError(f"Path does not exist: {codebase_path}")
    
    context = extract_codebase_context(codebase_path, max_files)
    return context

def query_min(query, type, context, codebase_path):
    matched_key, code, data = None, None, None
    if type == "function":
        keys = list(context['functions'].keys())
        matched_key = closest_match(query, keys)
        code = context['functions'][matched_key].copy()
    elif type == "class":
        keys = list(context['classes'].keys())
        matched_key = closest_match(query, keys)
        code = context['classes'][matched_key].copy()
    elif type == "module":
        if query in context["modules"]:
            file_path = context["modules"][query]
            matched_key = file_path
            data = context["file_structure"][file_path].copy()
            code = get_file_content(codebase_path, file_path)
        else:
            keys = list(context['modules'].keys())
            matched_key = closest_match(query, keys)
            file_path = context["modules"][matched_key]
            matched_key = file_path
            data = context["file_structure"][file_path].copy()
            code = get_file_content(codebase_path, file_path)
    else:
        keys = list(context['file_structure'].keys())
        matched_key = closest_match(query, keys, is_file=True)
        file_path = matched_key
        data = context["file_structure"][file_path].copy()
        code = get_file_content(codebase_path, file_path)

    if matched_key:
        if isinstance(code, Dict):
            code_lines = code['definition'].split('\n')
            if 'line_range' in code:
                line_range = code['line_range']
                for i in range(len(code_lines)):
                    code_lines[i] = f"{line_range[0]+i}\t{code_lines[i]}"
            code_str = '\n'.join(code_lines)
            code['definition'] = code_str
            return matched_key, code, data
        elif isinstance(code, str):
            code_lines = code.split('\n')
            line_range = (1, len(code_lines))
            for i in range(len(code_lines)):
                code_lines[i] = f"{line_range[0]+i}\t{code_lines[i]}"
            code_str = '\n'.join(code_lines)
            return matched_key, code_str, data

    return matched_key, code, data
        
def query_cumulative(queries, context, codebase_path):
    retrieved_data = {}

    results = {'file': {}, 'function': {}, 'class': {}}
    file_queries = [q['query'] for q in queries if q['type'] == 'file']
    module_queries = [q['query'] for q in queries if q['type'] == 'module']
    other_queries = [q['query'].split('@')[-1].strip() for q in queries if q['type'] == 'other' and '@' in q['query']]

    file_queries += other_queries
    
    for file_query in file_queries:
        file_match, code, file_data = query_min(file_query, 'file', context, codebase_path)
        if not file_match:
            continue
        results['file'][file_match] = (file_query, file_data)
        retrieved_data[file_query] = (file_match, code)

    for module_query in module_queries:
        file_match, code, file_data = query_min(module_query, 'module', context, codebase_path)
        if not file_match:
            continue
        results['file'][file_match] = (module_query, file_data)
        retrieved_data[module_query] = (file_match, code)

    for q in queries:
        if q['type'] in ['function', 'class']:
            matched_key, code, _ = query_min(q['query'], q['type'], context, codebase_path)
            if matched_key:
                if results['file']:
                    key = matched_key.split('.')[-1]
                    for file_data in results['file'].values():
                        if key not in file_data[1]['functions'] and key not in file_data[1]['classes']:
                            retrieved_data[q['query']] = (matched_key, code)
                else:
                    retrieved_data[q['query']] = (matched_key, code)

    return retrieved_data