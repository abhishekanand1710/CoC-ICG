sys_prompt = """
Act as an expert software developer who is really experienced in Python development.
You always generate correct code following the best practices and your code is always
thorough and handles all cases in order to be bug free. You can fix bugs given problem
statements by identifying the right changes to make in relevant code files.
"""

problem_staement_prompt = """
The following is the problem statement of an existing issue that is part of a repository
        
Issue: {problem}

Generate a git code patch in unified diff format to be applied to relevant files to fix the above issue.

Relevant code from the repository for the issues are given below:
{context}


"""

unified_diff_prompt = """
Given below are 2 example diffs for the same change.

Diff 1 - 

```diff
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
```

The following "high level diff" of the same
change is not as succinct as the minimal diff above,
but it is much easier to see two different coherent versions of the
`factorial()` function. Follow the below diff as a guideline 
on how to generate unified diffs for the problem statement.

Diff 2 - 

```diff
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
```
"""