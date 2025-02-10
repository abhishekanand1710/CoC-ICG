import os
import tiktoken
import argparse

def count_gpt_tokens(text, encoding_name="o200k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def analyze_repos(base_dir):
    text_extensions = {'.py'}

    models = {
        "gpt-4": "o200k_base"
    }

    for repo_name in os.listdir(base_dir):
        repo_path = os.path.join(base_dir, repo_name)
        if not os.path.isdir(repo_path):
            continue

        token_counts = {model: 0 for model in models}
        print(f"\nRepository: {repo_name}")

        for root, _, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext not in text_extensions:
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except (UnicodeDecodeError, IOError) as e:
                    continue

                for model, encoding_name in models.items():
                    try:
                        token_counts[model] += count_gpt_tokens(text, encoding_name)
                    except Exception as e:
                        print(f"  Error processing {model} tokens for {file_path}: {e}")

        print("Token counts:")
        for model, count in token_counts.items():
            print(f"  {model.upper():<10}: {count:,} tokens")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get GPT token counts for repositories")
    parser.add_argument("--repos_dir", type=str, help="Path to the base directory containing repositories")
    args = parser.parse_args()

    repos_dir = args.repos_dir

    analyze_repos(repos_dir)