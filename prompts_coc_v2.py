ANALYSIS_PROMPT = """You're a senior software engineer, debugging a given GitHub issue in a repository. Follow these steps:

1. Analyze the problem statement
2. Analyze the code that is already provided line by line and try to debug the issue.
3. Localize the issue at and determine how to fix it. Finally provide the code for the fix.
4. Proceed to solve it only when you are sure of the bug is present in the available context.
5. If the issue can't be debugged with the given code context, request SPECIFIC code elements using EXACT entity names or paths from the codebase.
  For eg - a particular function definition, class definition, variables, files or any other Pyhton entities.
6. Iteratively narrow down based on context.
7. Provide final solution as a git patch file.

## Issue:
{issue_description}

## Available Codebase Structure:
{file_structure}

## Current Context (Iteration {cur_iteration}):
{context_str}

## Guidelines:
- FIRST try to solve with existing context looking at each code line carefully.
- Check existing code and analyze what additional information is required to debug the issue and solve it.
- If the additional information required is present, then use it or else request for it.
- For requesting additional code, look at the issue and the provided code and think
  what extra information you need for finding the root cause of the issue.
- These are the different types of entities you can request - 
  1. FUNCTION - for functions
  2. CLASS - for classes
  3. FILE - for entire file
  4. OTHER - for any other data types or potential global variables or statements
- Be specific in your requests.
- Use the following format only for requesting code. 
  REQUESTS:
  NEED_CONTEXT: FUNCTION = <function_name> | Reason: <why is this required?>
  NEED_CONTEXT: FILE = <full_file_path> | Reason: <why is this required?>
  NEED_CONTEXT: CLASS = <class_name> | Reason: <why is this required?>
  NEED_CONTEXT: OTHER = <entity_type>@<file_path> | Reason: <why is this required?>

  Rules:
  1. Always use @<file_path> for OTHER type and don't use it for other types.
  2. One request per line.
  3. Only request FILE when you have the exact path or don't request without the file path.
  4. Maximum 3 context requests per analysis phase.
  5. Think carefully about what you want to request to solve the issue.
  6. Don't ask for code from your PREVIOUS REQUESTS that have already been found irrelevant.
  7. Don't use any extra words in the request lines.

## PREVIOUS REQUESTS:
{analysis_log}

- For solving, analyze the issue and relevant code and generate the correct SOLUTION as unified diff to fix the issue.
- Ensure that your fix will solve the original Issue and pass any potential test cases.
- Ensure proper indentation while generating the solution.
- Generate your solution as a single git patch file including all your file edits in the given format -
SOLUTION:
```diff
<code here>
```
  
"""


CONTEXT_TEMPLATE = """
#### Name: {name}
**File Path**: {file_path}

```python
{content}
```

Requested at iteration {iteration}
"""


FILTER_PROMPT = """You're a senior software engineer debugging a given GitHub issue in a repository.
You are provided with the issue and also code context fetched from the repository that are relevant.

## Current Issue:
{issue_description}

## Relevant Code:
{context_str}

Evaluate the candidate code given below to check if it's relevant to the issue and adds valuable 
information to already fetched code to solve the issue.

### Candidate Code:
**Reason behind requesting code:** {reason}
{candidate_context_str}

Respond in the following format:
IS_RELEVANT: Yes/No

If and only if the code contains more than a single class or function and there are irrelevant parts in it,
then extract the part you originally requested along with any other relevant code that might be useful.
Include the line numbers for each code line you extract and respond in the following format - 
IS_RELEVANT: Yes
RELEAVANT_CODE:
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

## Solution:
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

PATCH_GENERATION_PROMPT = """You're a senior software engineer, debugging a given GitHub issue in a repository.

## Issue:
{issue_description}

Given below is the solution along with your reasoning behind it for the issue.

## Solution:
# {solution}

## You want to edit the following files to fix the issue.
{patch_files}

### Files you have already edited to apply your fix - 
{edited_patch_files}

### You are currently editing file - `{current_file}`

## Editing phase
Here's all the relevant code you have found in file `{current_file}` to fix the issue.

### Relevant Code:
{cur_code_context}

### Edit Guidelines:
- Analyze your solution and relevant code from the file.
- Generate the new code along with the exact line numbers with proper indentation that you want to replace the original lines with.
- Only focus on the current file  - `{current_file}` and make necessary edits that are required in this file to fix the issue.
- You can make edits in 1 or multuiple places in the file.
- Return the complete range of lines that needs to edited and replaced or removed.
- Don't return line numbers in the code. Just return the code.
- Return your edits in the following format - 
EDIT CODE <start_line>,<end_line>:
```python
<new code here with exact line numbers or empty for removing lines>
```

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