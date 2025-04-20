ANALYSIS_PROMPT = """As a senior engineer, debug this GitHub issue. Follow steps:
1. Analyze problem statement
2. Line-by-line code review
3. Try to identify root cause location
4. If insufficient information, request SPECIFIC code functions/classes/files/variables using exact names/paths that are not present
5. Iteratively localize the issue by requesting unavailable code defiitions without repeating requests
6. Finally after gathering all context, output step-by-step root cause analysis analysis to find the EXACT bug and location.
7. Determine exact line/lines where the bug is and needs to be fixed

**Code Modules**: 
{modules}

**Rules**:
- Exhaust existing code before requesting new info
- Request missing critical entities (no functions/classes/files that are already present in the context)
- Request every possible entitiy or file related to the issue that are not available
- Differentiate between user's code entities in the issue and repository entities
- Only request entities that maybe present in the repository
- Entity types:
  1. FUNCTION: <name>
  2. CLASS: <name> 
  3. FILE: <full_path>
  4. OTHER: <type>@<file_path>
- Module requests: If uncertain about exact entities, request modules potentially containing them
- Format requests strictly as:
  NEED_CONTEXT: FUNCTION = <function_name> | Reason: <why is this required?>
  NEED_CONTEXT: FILE = <full_file_path> | Reason: <why is this required?>
  NEED_CONTEXT: CLASS = <class_name> | Reason: <why is this required?>
  NEED_CONTEXT: OTHER = <entity_type>@<file_path> | Reason: <why is this required?>
  NEED_MODULE: <exact_module_name> | Reason: <justification>
- Max 3 context/5 module requests per iteration
- Each request in new line
- If context gaps remain → MUST request first
- Always request with correct entity types, if not sure use OTHER
- Always use @<file_path> for OTHER type and don't use it for other types
- No partial solutions allowed and only provide necessary changes

## Issue: 
{issue_description}

## Your previous analysis and all the code you requested:

{context_str}

**Previous Requests**:
{analysis_log}


**Output either**:
Your updated line by line analysis of the issue with all the previous and new code you have and new context/module requests that have not been analyzed in the given format below.
You may need to correct your previous analysis.

LINE-BY-LINE ANALYSIS:
1. <analysis>
2. <analysis>
...

NEW REQUESTS:
<requests>

**OR** when all relevant code is fetched for debugging, output

ROOT CAUSE ANALYSIS:
<step-by-step breakdown>

{iteration_check_statement}"""

SOLVE_PROMPT= """As a senior engineer write the fix for the Github issue. Follow steps:
1. Analyze the issue clearly
2. Line-by-line code review of all the provided code
3. Line-by-line review of your analysis while fetching relevant code for the issue
3. Generate ste-by-step root cause analysis of the issue to determine the exact location of the bug
4. Generate code fix for the issue

**Issue**:
{issue_description}

**Code Context**:
{context_str}

**Your previous analysis**:
{analysis}

## Guidelines:
- Generate your new root cause analysis first to localize the issue
- Apply your root cause analysis while generating fix for the issue
- Maintain code style/indentation
- Edit minimal required files
- Generate your solution as a single git patch file including all your file edits in the given format -
SOLUTION:
```diff
<code here>
```"""

ITER_CONTEXT_TEMPLATE = """
**Analysis at iteration {iteration}**:
{analysis}

**Code fetched after analysis**:
{code}

"""

CONTEXT_TEMPLATE = """
**File Path**: {file_path}
**Reason behind requesting the code:** {reason}

```python
{content}
```
"""

CANDIDATE_CONTEXT_TEMPLATE = """
#### Name: {name}
**Reason behind requesting the code:** {reason}
**File Path**: {file_path}

```python
{content}
```

Requested at iteration {iteration}
"""

FILTER_PROMPT = """As a senior engineer, you are debugging a given GitHub issue.
You are provided with the issue and also existing code fetched from the repository that is relevant.
You are assessing the relevance of a piece of candidate code to the issue.

**Issue**: 
{issue_description}

**Existing Code**:
{context_str}

**Candidate Code**:
{candidate_context_str}

1. Verify if candidate code contains requested entities/patterns
2. Look at the reason you requested the code.
3. Line-by-line by code review of candidate code.
4. Check value-added beyond existing code.
5. If >1 class/function, extract all relevant lines with line numbers from the code.

RULES:
- Answer strictly in format:
IS_RELEVANT: Yes/No
(Only if Yes with extraction needed)
RELEVANT_CODE:
```python
<code here>
```
"""

REQUEST_FILES_TO_BE_EDITED_PROMPT = """You're a senior software engineer, debugging a given GitHub issue in a repository.

## Issue:
{issue_description}

Given below is all the relevant code you have found that is 
required to debug the issue and the solution you generated for it.

## Relevant Code:
{code_context}

## Solution that you have generated:
{solution}

Now, looking at the solution, return the list of files that you need to edit to fix the issue.
Only return the files that are used in the solution to fix the issue.
Return the files in the following format - 
FILE_REQUEST: <path_to_file>

Rules:
1. Each file request should start in a new line.
2. There should be no extra words in the line.
3. Return only the file or files that needs to be edited.
4. Return the full path of the file.

Example:
FILE_REQUEST: src/model.py
"""

PATCH_GENERATION_PROMPT = """As a senior engineer, fixing a GitHub issue, follow the steps:
1. Analyze the issue
2. Read the solution line by line

**Issue:**  
{issue_description}  

**Solution:**  
{solution}  

**File to be edit for the soluton**:
{patch_files}

**Already edited:**
{edited_patch_files}

**Currently editing**: `{current_file}`

**Relevant code in `{current_file}`**: 
{cur_code_context}

**Edit Guidelines:**  
1. Analyze the solution and relevant code.  
2. Generate new code (with proper indentation) to replace/remove/insert lines in `{current_file}`.  
3. Focus only on necessary edits in this file (1+ locations).  
4. Return complete line ranges (start-end) for each edit.  
5. **Do NOT include line numbers in the code block.**  
6. Return your edits in the following format - 
EDIT CODE <start_line>,<end_line>:
```python
<new code here with exact line numbers or empty for removing lines>
```

**Examples**:
Example for replacing a single line with one line:
EDIT CODE 45,45:
```python
def call_service(service=None, min_n=10):
```

Example for inserting 2 lines after a line 44. Giving <start_line> greater than <end_line> inserts new line after <end_line>:
EDIT CODE 45,44:
```python
def check():
  print("Checking")
```

Example for replacing 3 lines with 2 lines as range is 48-50 and only 2 lines are provided:
EDIT CODE 48,50:
```python
  if count < min_n:
    return
```

Example for removing 2 lines
EDIT CODE 101,102:
```python
```      
"""