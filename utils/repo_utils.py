import git
from pathlib import Path
import os
import subprocess
import tempfile
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def checkout_git_repo_at_commit(base_dir, repo_name, commit):
    '''
    Checks out git repo at the specified base commit of the instance in the dataset
    '''
    instance_repo = Path(base_dir, repo_name)
    if not instance_repo.exists():
        repo = git.Repo.clone_from(f"https://github.com/{repo_name}.git", instance_repo.resolve())
    else:
        repo = git.Repo(instance_repo.resolve())
    repo.git.reset('--hard')
    repo.git.clean('-fd')
    repo.git.checkout(commit)
    repo.close()
    
    return instance_repo

def find_non_utf_encoded_files_in_dir(dir):
    '''
    Returns a list of files that are not utf-8 encoded in a repository
    '''
    invalid_files = []
    for root, _, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    f.read()
            except UnicodeDecodeError:
                invalid_files.append(file_path)
    return invalid_files

def init_vectorstore(instance_id, splits, embeddings_dir):
    '''
    Initialized a vectorstore for repo code embeddings from local storage if embeddings are already generated
    or from code files.
    '''
    embeddings_path = f"{embeddings_dir}/{instance_id}"
    if os.path.exists(embeddings_path):
        vectorstore = FAISS.load_local(embeddings_path, 
                                       OpenAIEmbeddings(model="text-embedding-3-small"), 
                                       allow_dangerous_deserialization=True)
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(embeddings_path)

    return vectorstore.as_retriever(search_kwargs={"k": 20})

def process_repo(instance, base_dir, embeddings_dir):
    '''
    Prepares the instance repository for inference by checking it out at the base commit
    and initialize a retriever from an embeddings vectorstore
    '''
    repo_dir = checkout_git_repo_at_commit(base_dir, repo_name=instance['repo'], commit=instance['base_commit'])

    non_utf_files = find_non_utf_encoded_files_in_dir(repo_dir)
    loader = GenericLoader.from_filesystem(
        repo_dir,
        glob="**/*",
        exclude=non_utf_files,
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
        # show_progress=True
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=0
    )
    splits = text_splitter.split_documents(documents)
    vectorstore = init_vectorstore(instance['instance_id'], splits, embeddings_dir)
    return vectorstore

def process_repo_min(instance, base_dir: str):
    '''
    Prepares the instance repository for inference by checking it out at the base commit
    '''
    repo_dir = checkout_git_repo_at_commit(base_dir, repo_name=instance['repo'], commit=instance['base_commit'])
    return repo_dir

def apply_edits(file_path, edits):
    '''
    Takes in file path and a list of edits to apply and returns the new file content after applying the edits
    '''
    with open(file_path, 'r') as f:
        content = f.read().split('\n')
    
    for start, end, new_lines in sorted(edits, reverse=True):
        if start <= end:  # replace/remove
            start_idx = start - 1
            end_idx = end
            if new_lines:
                content[start_idx:end_idx] = [new_lines]
            else:
                del content[start_idx:end_idx]
        else:  # insert (start > end)
            insert_pos = end
            content[insert_pos:insert_pos] = [new_lines]
    
    return '\n'.join(content)

def generate_patch_file(file_edits, codebase_path):
    '''
    Generates a git patch file by applying all file edits to the codebase
    '''
    repo_path = os.path.abspath(codebase_path)
    orig_dir = os.getcwd()

    try:
        os.chdir(repo_path)
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if status.stdout.strip():
            raise ValueError("Repo has uncommitted changes")
        
        patch_file = tempfile.NamedTemporaryFile(delete=False, suffix='.patch')
        patch_file.close()

        old_file_contents = {}
        for file, edits in file_edits.items():
            file_path = f"{repo_path}/{file}"

            with open(file_path, 'r') as f:
                old_file_content = f.read()
            new_file_content = apply_edits(file_path, edits)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)

            old_file_contents[file_path] = old_file_content

        subprocess.run(['git', 'diff', '--no-color'], stdout=open(patch_file.name, 'w'), check=True)
        with open(patch_file.name, 'r', encoding='utf-8') as f:
            patch_content = f.read()

        for file_path, content in old_file_contents.items():
            full_path = os.path.join(repo_path, file_path)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        os.unlink(patch_file.name)
        return patch_content
    finally:
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_path, stdout=subprocess.DEVNULL)
        os.chdir(orig_dir)
    


    



