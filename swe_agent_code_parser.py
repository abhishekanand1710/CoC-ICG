import os
import json
from typing import Dict, List, Optional, Any
from utils.repo_utils import *
from tree_sitter import Language, Parser
import tree_sitter_python
import difflib

PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

def extract_codebase_context(codebase_path: str, max_files: Optional[int] = None) -> Dict[str, Any]:
    context = {
        "modules": [],
        "classes": {},
        "functions": {},
        "imports": {},
        "file_structure": {},
        "documentation": {},
        "dependencies": []
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
    
    import_query = PY_LANGUAGE.query("""
        (import_statement
          (dotted_name) @module_name)
        
        (import_from_statement
          module_name: (dotted_name) @from_module
          name: (dotted_name) @imported_name)
    """)

    global_query = PY_LANGUAGE.query("""
        (module
          (expression_statement
            (assignment
              left: [
                (identifier) @global_var_name
                (pattern_list
                  (identifier) @global_var_name)
              ]
              right: (_) @global_var_value)) @global_assignment
              
          (expression_statement) @global_statement
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
        if 'tests/' in file_path:
            continue
        rel_path = os.path.relpath(file_path, codebase_path)
        module_name = rel_path.replace('/', '.').replace('\\', '.').replace('.py', '')
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            # try:
            code = file.read()
            tree = parser.parse(bytes(code, 'utf-8'))
            root_node = tree.root_node
            
            file_info = {
                "path": rel_path,
                "module_name": module_name,
                "functions": [],
                "classes": [],
                "global": [],
                "docstring": extract_module_docstring(root_node, code)
            }

            for match in function_query.matches(root_node):
                function_info = extract_function_info(match, code)
                function_info["file_path"] = rel_path
                file_info["functions"].append(function_info["name"])
                context["functions"][f"{module_name}.{function_info['name']}"] = function_info

            for match in class_query.matches(root_node):
                class_info = extract_class_info(match, code)
                class_info["file_path"] = rel_path
                file_info["classes"].append(class_info["name"])
                context["classes"][f"{module_name}.{class_info['name']}"] = class_info
            
            # for match in global_query.matches(root_node):
            #     print(rel_path)
            #     global_info = extract_global_info(match, code)
            #     global_info["file_path"] = rel_path
            #     file_info["global"].append(global_info["file_path"])
            #     context["global"][f"{module_name}.{global_info['name']}"] = class_info
            
            # # imports
            # for match in import_query.matches(root_node):
            #     import_info = extract_import_info(match, code)
            #     file_info["imports"].extend(import_info)
            #     for imp in import_info:
            #         if imp not in context["imports"]:
            #             context["imports"][imp] = []
            #         context["imports"][imp].append(module_name)
                    
            #         # # Track external dependencies
            #         # if not is_standard_library(imp.split('.')[0]):
            #         #     if imp.split('.')[0] not in context["dependencies"]:
            #         #         context["dependencies"].append(imp.split('.')[0])
            
            context["modules"].append(file_info)
            context["file_structure"][rel_path] = file_info
            
            if file_info["docstring"]:
                context["documentation"][module_name] = file_info["docstring"]
    
    return context

def extract_module_docstring(node, code):
    if node.children and node.children[0].type == 'module':
        for child in node.children[0].children:
            if child.type == 'string':
                return code[child.start_byte:child.end_byte].strip('\'\"')
    return None

def extract_function_info(match, code):
    match = match[1:]
    code = code.split('\n')

    function_name_node = next((node['function_name'] for node in match if 'function_name' in node), None)
    params_node = next((node['params'] for node in match if 'params' in node), None) 
    body_node = next((node['body'] for node in match if 'body' in node), None)

    start_row , start_col= function_name_node[0].start_point
    end_row, end_col = function_name_node[0].end_point
    function_name = code[start_row:end_row+1][0][start_col:end_col]

    params_end_row= params_node[0].end_point.row
    function_sig = code[start_row:params_end_row+1]
    function_sig = '\n'.join(function_sig)
    
    body_end_row= body_node[0].end_point.row
    function_def = code[start_row:body_end_row+1]
    function_def = '\n'.join(function_def)

    return {
        "name": function_name,
        "signature": function_sig,
        "definition": function_def
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
        "definition": class_def
    }

def extract_global_info(match, code):
    code = code.split('\n')
    print(match)

def extract_import_info(match, code):
    imports = []
    match = match[1:]
    code = code.split('\n')

    module_node = next((node['module_name'] for node in match if 'module_name' in node), None)
    from_module_node = next((node['from_module'] for node in match if 'from_module' in node), None)
    imported_name_node = next((node['imported_name'] for node in match if 'imported_name' in node), None)
    
    if module_node:
        start_row , _ = module_node[0].start_point
        end_row, _ = module_node[0].end_point
        module_name = code[start_row:end_row+1][0]
        imports.append(module_name)
    
    if from_module_node and imported_name_node:
        start_row , _ = from_module_node[0].start_point
        end_row, _ = from_module_node[0].end_point
        from_module = code[from_module_node[0].start_point.row:from_module_node[0].end_point.row+1][0]
        imports.append(from_module)
    
    return imports

def get_file_content(codebase_path, file_path):
    full_path = os.path.join(codebase_path, file_path)
    print(full_path)
    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
    
def closest_match(target, string_list):
    original_strings = string_list.copy()
    processed_strings = [s.split('.')[-1] for s in string_list]
    closest = difflib.get_close_matches(target, processed_strings, n=1, cutoff=0)[0]
    match_idx = processed_strings.index(closest)
    return original_strings[match_idx]

def analyze_codebase(codebase_path, output_path='file_structure.json', max_files=None):
    if not os.path.exists(codebase_path):
        raise ValueError(f"Path does not exist: {codebase_path}")
    
    context = extract_codebase_context(codebase_path, max_files)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(context['file_structure'], f, indent=2)
    
    return context
    
def query_context(query, type, context, codebase_path):
    if type == "function":
        keys = list(context['functions'].keys())
        matched_key = closest_match(query, keys)
        return matched_key, context['functions'][matched_key]
    elif type == "class":
        keys = list(context['classes'].keys())
        matched_key = closest_match(query, keys)
        return matched_key, context['classes'][matched_key]
    else:
        for module in context["modules"]:
            if module["path"].endswith(query) or query in module["path"]:
                return module["path"], get_file_content(codebase_path, module["path"])
            

context = analyze_codebase('swe_bench_cache/repos/astropy/astropy/astropy/modeling')
# print(query_context('_separable', 'function', context, 'blah'))