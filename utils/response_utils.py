import re

def extract_patch_from_markdown(text):
    import re
    pattern = r'```(?:patch|diff)\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).rstrip() if match else None