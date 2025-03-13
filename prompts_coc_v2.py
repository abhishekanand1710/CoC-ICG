ANALYSIS_PROMPT = """You're a senior software engineer, debugging a given GitHub issue in a repository. Follow these steps:

1. Analyze the problem statement
2. Analyze the code that is already provided line by line and try to debug the issue.
3. Localize the issue at and determine the complete root cause.
4. If the issue can't be debugged with the given code context, request SPECIFIC code elements using EXACT entity names or paths from the codebase.
  For eg - a particular function definition, class definition, variables, files or any other Pyhton entities.
5. Iteratively narrow down based on context.
6. Provide final step by step analysis of the issue based on the issue description and code from the repository.

## Issue:
{issue_description}

## Codebase moduless:
{modules}

## Current Context (Iteration {cur_iteration}):
{context_str}

## PREVIOUS REQUESTS:
{analysis_log}

## Guidelines:
- Check existing code line by line to debug the issue.
- Analyze what additional information is required to debug the issue and find its root cause.
- First find the required information in the code provided and use it.
- If the additional information required is not present then request for it.
- Always try to find all the information required to fix the issue without making assumptions.
- For requesting additional code, look at the issue and the provided code and think
  what extra information you need for finding the root cause of the issue.
- Differentiate between what are user's code entities given in the issue and what are repository entities.
  Only request entities that maybe present in the repository as that would help you solve the issue.
- These are the different types of entities you can request - 
  1. FUNCTION - for specific functions present in the repository
  2. CLASS - for specific classes present in the repository
  3. FILE - for entire file present in the repository for complete analysis or when function/class relevant are not known.
  4. OTHER - for any other data types or potential global variables or statements
- Be specific in your requests.
- Use the following format only for requesting code. 
  REQUESTS:
  NEED_CONTEXT: FUNCTION = <function_name> | Reason: <why is this required?>
  NEED_CONTEXT: FILE = <full_file_path> | Reason: <why is this required?>
  NEED_CONTEXT: CLASS = <class_name> | Reason: <why is this required?>
  NEED_CONTEXT: OTHER = <entity_type>@<file_path> | Reason: <why is this required?>
- If you are not sure about the exact entities that may contain the issue - Request 5 modules from the module list 
  to identify files that may contain the issue.
- Make your request in the following format - 
  NEED_MODULE: <exact module name from modules list> | Reason: <why this module is required?>
  Rules:
  1. Always use @<file_path> for OTHER type and don't use it for other types.
  2. One request per line.
  3. Only request FILE when you have the exact path or don't request without the file path.
  4. Maximum 3 context requests or 5 module requests per analysis phase.
  5. Think carefully about what you want to request to solve the issue.
  6. Don't ask for code from your PREVIOUS REQUESTS that have already been found irrelevant.
  7. Don't use any extra words in the request lines.
  8. Each module request should also begin in a new line and use the exact module name.

- For solving, analyze the issue and all the relevant code fetched from the repository
  and provide a complete step by step breakdown of the issue below.
ROOT CAUSE ANALYSIS:
<analysis here>

Now, respond with either your NEED_CONTEXT requests that are new and not in PREVIOUS REQUESTS or your detailed ROOT CAUSE ANALYSIS.
{iteration_check_statement}
"""


ANALYSIS_PROMPT_V2 = """As a senior engineer, debug this GitHub issue. Follow steps:
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

**Your previous analysis**:
{previous_analysis}

**Your requested code**: 
{context_str}

**Previous Requests**:
{analysis_log}


**Output either**:
new context/module requests that have not been analyzed in the following format:

Line-by-line analysis of relevant code:
<analysis>

NEW REQUESTS:
<requests>

**OR** when all relevant code is fetched for debugging, output

ROOT CAUSE ANALYSIS:
<step-by-step breakdown>

{iteration_check_statement}"""


TEST_PROMPT = """As a senior engineer, who is debugging this GitHub issue, write test cases for issue verification. Follow these steps:
1. Analyze issue
2. Line-by-line code review of provided code
3. Line-by-line review of root cause analysis
4. Generate test cases for issue fix verification

**Issue**:
{issue_description}

**Code Context**:
{context_str}

**Root Cause Analysis**:
{analysis}

**Requirements**: 
1. Generate reproduction test cases matching root cause  
2. Create validation tests for potential fixes  
3. Multiple test cases allowed
- Return your test cases in the following format:
TEST CASE:
```python
<code here>
```
"""

SOLVE_PROMPT_WITH_TESTS = """As a senior engineer write the fix for the Github issue. Follow steps:
1. Analyze issue
2. Line-by-line code review of provided code
3. Line-by-line review of root cause analysis
4. Line-by-line review of test cases.
5. Generate code fix for the issue.

**Issue**:
{issue_description}

**Code Context**:
{context_str}

**Root Cause Analysis**:
{analysis}

**Test Cases**:
{test_cases}

## Guidelines:
- Apply root cause analysis from above
- Ensure fix addresses all test cases
- Maintain code style/indentation
- Edit minimal required files
- Generate your solution as a single git patch file including all your file edits in the given format -
SOLUTION:
```diff
<code here>
```"""

SOLVE_AND_ANALYZE_PROMPT= """As a senior engineer write the fix for the Github issue. Follow steps:
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

SOLVE_PROMPT= """As a senior engineer write the fix for the Github issue. Follow steps:
1. Analyze issue
2. Line-by-line code review of provided code
3. Line-by-line review of root cause analysis
4. Generate code fix for the issue.

**Issue**:
{issue_description}

**Code Context**:
{context_str}

**Root Cause Analysis**:
{analysis}

## Guidelines:
- Apply root cause analysis from above while generating the fix
- Maintain code style/indentation
- Edit minimal required files
- Generate your solution as a single git patch file including all your file edits in the given format -
SOLUTION:
```diff
<code here>
```"""

CONTEXT_TEMPLATE = """
**File Path**: {file_path}

```python
{content}
```

Requested at iteration {iteration}
"""

CONTEXT_TEMPLATE_EXTN = """
**File Path**: {file_path}

**Functions and classes in file**:
{additional_content}

```python
{content}
```

Requested at iteration {iteration}
"""

CANDIDATE_CONTEXT_TEMPLATE = """
#### Name: {name}
**Reason behind requesting code:** {reason}
**File Path**: {file_path}

```python
{content}
```

Requested at iteration {iteration}
"""


FILTER_PROMPT = """You're a senior software engineer debugging a given GitHub issue in a repository.
You are provided with the issue and also code context fetched from the repository that are relevant.

## Issue:
{issue_description}

## Relevant Code:
{context_str}

Evaluate the candidate code given below to check if it's relevant to the issue and adds valuable 
information to already fetched code to solve the issue.

### Candidate Code:
{candidate_context_str}

Think about the reason you requested the code and respond whether it's relevant or not in the following format:
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

FILTER_PROMPT_V2 = """As a senior engineer, you are debugging a given GitHub issue.
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
2. Read the solutio line by line

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