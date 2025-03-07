import git
from pathlib import Path
import os

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_file_content(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"File '{file_path}' does not exist"
    
    if os.path.isdir(file_path):
        return f"'{file_path}' is a directory, not a file"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        return f"File '{file_path}' is not a text file or contains non-UTF-8 characters"

def checkout_git_repo_at_commit(base_dir, repo_name, commit):
    instance_repo = Path(base_dir, repo_name)

    if not instance_repo.exists():
        repo = git.Repo.clone_from(f"https://github.com/{repo_name}.git", instance_repo.resolve())
    else:
        repo = git.Repo(instance_repo.resolve())

    repo.git.checkout(commit)
    repo.close()
    
    return instance_repo

def find_non_utf_encoded_files_in_dir(dir):
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

def init_vectorstore(instance_id, splits):
    embeddings_path = f"swe_bench_cache/repo_embeddings/{instance_id}"
    if os.path.exists(embeddings_path):
        vectorstore = FAISS.load_local(embeddings_path, 
                                       OpenAIEmbeddings(model="text-embedding-3-small"), 
                                       allow_dangerous_deserialization=True)
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(embeddings_path)

    return vectorstore

def process_repo(instance, base_dir: str):
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
    vectorstore = init_vectorstore(instance['instance_id'], splits)
    return vectorstore

def process_repo_min(instance, base_dir: str):
    repo_dir = checkout_git_repo_at_commit(base_dir, repo_name=instance['repo'], commit=instance['base_commit'])
    return repo_dir