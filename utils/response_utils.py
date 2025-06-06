import re

def extract_patch_from_markdown(text):
    '''
    Extracts patch from model response
    '''
    pattern = r'```(?:patch|diff)\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).rstrip() if match else None

def parse_analysis_response(response):
    '''
    Extracts context requests if they exist from the analysis stage response or returns
    '''
    if 'NEED_CONTEXT' in response or 'NEED_MODULE' in response:
        response_lines = response.split('\n')
        req_functions = []
        req_classes = []
        req_files = []
        req_modules = []
        others = {}

        line_by_line_analysis_start_idx = None
        line_by_line_analysis_end_idx = None

        for idx, line in enumerate(response_lines):
            line = line.strip()
            if 'LINE-BY-LINE ANALYSIS' in line:
                line_by_line_analysis_start_idx = idx + 1
                continue
            if 'NEW REQUESTS' in line:
                line_by_line_analysis_end_idx = idx
                continue
            try:
                if 'NEED_CONTEXT' in line:
                    line, reason = line.split('|', 1)
                    line = line.strip()
                    if 'FUNCTION' in line:
                        func_name = line.split(' ')[-1].split('@')[0]
                        req_functions.append((func_name, reason))
                    elif 'CLASS' in line:
                        class_name = line.split(' ')[-1].split('@')[0]
                        req_classes.append((class_name, reason))
                    elif 'FILE' in line:
                        file = line.split(' ')[-1]
                        req_files.append((file, reason))
                    elif 'OTHER' in line:
                        req = line.strip().split(' ')[-1].split('@')
                        req = [v.strip() for v in req]
                        others[req[0]] = (req[1], reason)
                elif 'NEED_MODULE' in line:
                    line, reason = line.split('|', 1)
                    module_name = line.split(':')[-1].strip()
                    req_modules.append((module_name, reason))
            except: continue
        
        line_by_line_analysis = None
        if line_by_line_analysis_start_idx and line_by_line_analysis_end_idx:
            line_by_line_analysis = '\n'.join(response_lines[line_by_line_analysis_start_idx:line_by_line_analysis_end_idx]).strip()

        return True, {
            "functions": req_functions,
            "classes": req_classes,
            "files": req_files,
            "others": others,
            "modules": req_modules,
            "line_by_line_analysis": line_by_line_analysis
        }
    else:
        return False, response

def parse_file_request_response(response):
    '''
    Parses a file request response to return the list of files requested for editing
    '''
    response_lines = response.split("\n")
    file_requests = []
    for line in response_lines:
        line = line.strip()
        if "FILE_REQUEST"  in line:
            file_path = re.search(r'FILE_REQUEST:\s*([\w/\.]+)', line)
            if file_path:
                file_requests.append(file_path.group(1))
    return file_requests

def parse_filter_response(response):
    '''
    Parses a filter stage response and returns whether retrieved code is relevant to the issue.
    '''
    response_lines = response.split("\n")

    is_relevant = False
    selected_code_present = False
    code_idx = None
    for i, line in enumerate(response_lines):
        line = line.strip()
        if "IS_RELEVANT" in line:
            if "no" in line.lower():
                return False, None
            if "yes" in line.lower():
                is_relevant = True
        if is_relevant and line.startswith('RELEVANT_CODE'):
            selected_code_present = True
            code_idx = i + 1
            break
    
    if selected_code_present and code_idx:
        code_block = '\n'.join(response_lines[code_idx:]).strip()
        # Remove any code blocks at start/end
        pattern = r'```python\n(.*?)```'
        match = re.search(pattern, code_block, re.DOTALL)
        if match:
            return is_relevant, match.group(1).strip()
        return is_relevant, None
    
    return is_relevant, None

def parse_file_edit_response(response):
    '''
    Parses edit stage response to return file edit data containing line numbers and code snippets
    '''
    pattern = r'EDIT CODE (\d+),(\d+):\s*```python\n(.*?)```'
    matches = re.finditer(pattern, response, re.DOTALL)
    code_edit_snippets = []
    for match in matches:
        start_line = int(match.group(1))
        end_line = int(match.group(2))
        code = match.group(3)
        code_edit_snippets.append((start_line, end_line, code))
    return code_edit_snippets