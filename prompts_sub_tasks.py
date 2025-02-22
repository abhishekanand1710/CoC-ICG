sys_prompt = """
Act as an expert software developer who is really experienced in Python development.
You always generate correct code following the best practices and your code is always
thorough and handles all cases in order to be bug free. You can fix bugs given problem
statements by identifying the right changes to make in relevant code files.
"""

diff_patch_example = """<patch>
diff --git a/... b/...
--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
 def euclidean(a, b):
-    while b:
-        a, b = b, a % b
-    return a
+    if b == 0:
+        return a
+    return euclidean(b, a % b)
 
 
 def bresenham(x0, y0, x1, y1):
     points = []
     dx = abs(x1 - x0)
     dy = abs(y1 - y0)
-    sx = 1 if x0 < x1 else -1
-    sy = 1 if y0 < y1 else -1
-    err = dx - dy
+    x, y = x0, y0
+    sx = -1 if x0 > x1 else 1
+    sy = -1 if y0 > y1 else 1
 
-    while True:
-        points.append((x0, y0))
-        if x0 == x1 and y0 == y1:
-            break
-        e2 = 2 * err
-        if e2 > -dy:
+    if dx > dy:
+        err = dx / 2.0
+        while x != x1:
+            points.append((x, y))
             err -= dy
-            x0 += sx
-        if e2 < dx:
-            err += dx
-            y0 += sy
+            if err < 0:
+                y += sy
+                err += dx
+            x += sx
+    else:
+        err = dy / 2.0
+        while y != y1:
+            points.append((x, y))
+            err -= dx
+            if err < 0:
+                x += sx
+                err += dy
+            y += sy
 
+    points.append((x, y))
     return points
</patch>"""

problem_statement_prompt = """
You will be provided with a partial code base that you have never seen before 
and an issue statement explaining a bug in the codebase that needs to be resolved.
        
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
Solve the issue by breaking the process into sequential sub-tasks and generating code for each
sub-task wherever required. Remember you can not execute code and hence, don't write unit tests.
Determine the root cause of the issue and fix it by making appropriate changes to the code. 
Perform your actions by breaking them into sub-tasks.

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