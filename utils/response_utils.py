import re

def extract_diff_from_markdown(text):
    import re
    pattern = r'```diff\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).rstrip() if match else None