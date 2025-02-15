import git
from pathlib import Path

def checkout_git_repo_at_commit(base_dir, repo_name, commit):
    repo_dir = f"{repo_name}.git"

    instance_repo = Path(base_dir, repo_name)

    if not instance_repo.exists():
        repo = git.Repo.clone_from(f"https://github.com/{repo_name}.git", instance_repo.resolve())
    else:
        repo = git.Repo(instance_repo.resolve())

    repo.git.checkout(commit)
    repo.close()
    
    return instance_repo