sys_prompt = """
Act as an expert software developer who is really experienced in Python development.
You always generate correct code following the best practices and your code is always
thorough and handles all cases in order to be bug free. You can fix bugs given problem
statements by identifying the right changes to make in relevant code files.
"""

problem_statement_prompt = """
You will be provided with a partial code base and an issue statement explaining a problem to resolve.
        
<issue>
{problem}
</issue>

<code>
{context}
</code>

Here is an example of a patch file. It consists of changes to the code base. 
It specifies the file names, the line numbers of each change, and the removed and added lines. 
A single patch file can contain changes to multiple files.
"""

final_inference_prompt = """
I need you to solve the provided issue by breaking down the problem into sub-tasks and solving each
sub-task one by one by generating the code wherever required. Don't include unit tests in your sub-tasks.
Just focus on solving the issue.

Finally, generate a single patch file by combining the solution for each sub-task that I can apply 
directly to this repository using git apply. Each patch file can contain changes to multiple 
files to fix the issue. Please respond with all the sub-tasks and the final patch file in the format 
shown above. The patch file should just focus on fixing the issue without generating any additional
code for any tests or other things.

Respond below:
"""

relevant_code_snippet_prompt = """
<filepath>
{filepath}
</filepath>

<code_content>
{code}
</code_content>
"""

unified_diff_prompt = """<patch_guidleine>
Given below are 2 example diffs for the same change.

Diff 1 - 

<dispreferred_patch>
diff --git a/... b/...
@@ ... @@
-def factorial(n):
+def factorial(number):
-    if n == 0:
+    if number == 0:
         return 1
     else:
-        return n * factorial(n-1)
+        return number * factorial(number-1)
</dispreferred_patch>

The following "high level diff" of the same
change is not as succinct as the minimal diff above,
but it is much easier to see two different coherent versions of the
`factorial()` function. Follow the below diff as a guideline 
on how to generate unified diffs for the problem statement.

Diff 2 - 

<preferred_patch>
diff --git a/... b/...
@@ ... @@
-def factorial(n):
-    if n == 0:
-        return 1
-    else:
-        return n * factorial(n-1)
+def factorial(number):
+    if number == 0:
+        return 1
+    else:
+        return number * factorial(number-1)
</preferred_patch>
</patch_guidleine>
"""