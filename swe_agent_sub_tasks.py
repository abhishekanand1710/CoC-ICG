import os
from typing import TypedDict, List, Optional
import json
from tqdm import tqdm
import argparse
import traceback
import glob

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_text_splitters import Language
import langchain

from utils.git_utils import checkout_git_repo_at_commit
from utils.response_utils import extract_patch_from_markdown
from prompts_sub_tasks import *

langchain.verbose = True

llm = None

class GraphState(TypedDict):
    instance_id: str
    problem: str
    instance_repo: str
    documents: Optional[List[str]] = None
    context: Optional[str] = None
    patch: Optional[str] = None

def init_backend(model):
    if model in ['gpt-4o-mini']:
        return ChatOpenAI(model=model)
    else:
        raise NotImplementedError(f"Support for {model} is not yet implemented.")
    
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
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

def create_retriever(instance_id, splits):
    embeddings_path = f"swe_bench_cache/repo_embeddings/{instance_id}"
    if os.path.exists(embeddings_path):
        vectorstore = FAISS.load_local(embeddings_path, 
                                       OpenAIEmbeddings(model="text-embedding-3-small"), 
                                       allow_dangerous_deserialization=True)
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(embeddings_path)

    return vectorstore.as_retriever(search_kwargs={"k": 20})

def retrieve_documents(state: GraphState):
    retriever = create_retriever(state["instance_id"], state["documents"])
    relevant_docs = retriever.invoke(state["problem"])

    context = "\n".join([relevant_code_snippet_prompt.format(filepath = d.metadata['source'][len(state["instance_repo"]):], code = d.page_content) for d in relevant_docs])
    return {"context": context}

def generate_patch(state: GraphState):

    inference_prompt = f"{sys_prompt}\n{problem_statement_prompt}\n{unified_diff_prompt}\n{final_inference_prompt}"
    prompt = ChatPromptTemplate.from_template(
        inference_prompt
    )
    
    chain = prompt | llm
    
    response = chain.invoke({
        "problem": state["problem"],
        "context": state["context"]
    })
    
    return {"patch": response.content}

def build_workflow():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_patch)
    

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


def run_inference(instance, base_repo_dir):
    splits = process_repo(instance, base_repo_dir)

    initial_state = GraphState(
        instance_id=instance["instance_id"],
        instance_repo=f"{base_repo_dir}/{instance['repo']}/",
        problem=instance["problem_statement"],
        documents=splits
    )

    app = build_workflow()
    result = app.invoke(initial_state)
    patch = extract_patch_from_markdown(result["patch"])

    return result["patch"], patch

def load_processed_instances(output_file):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            processed_instances = {entry["instance_id"]: entry for entry in json.load(f)}
        return processed_instances
    return {}

def filter_out_processed_instances(dataset, processed_instances):
    dataset = {key: value for key, value in dataset.items() if key not in processed_instances}
    return dataset

def load_local_dataset(input_file, processed_instances):
    with open(input_file, "r") as f:
        dataset = json.load(f)
    
    dataset = filter_out_processed_instances(dataset, processed_instances)
    return dataset

def main(args):
    run_id = args.run_id
    if not os.path.exists(f"output/{run_id}"):
        os.mkdir(f"output/{run_id}")

    output_file = f"output/{run_id}/prediction_rag_sub_tasks_{args.model}.json"
    responses_file = f"output/{run_id}/response_rag_sub_tasks_{args.model}.json"
    processed_instances = load_processed_instances(output_file)
    responses = load_processed_instances(responses_file)

    dataset = load_local_dataset(args.input_file, processed_instances)
    print(f"Generating for {len(dataset)} samples.")
    
    preds = [value for _, value in processed_instances.items()]
    cur_responses = [value for _, value in responses.items()]
    try:
        for key, test_instance in tqdm(dataset.items()):
            result, patch = run_inference(test_instance, args.dir)
            output = {
                "instance_id": key,
                "model_patch": patch,
                "model_name_or_path": args.model
            }
            preds.append(output)

            response_output = {
                "instance_id": key,
                "response": result,
                "model_name_or_path": args.model
            }
            cur_responses.append(response_output)

            if len(preds)%5 == 0:
                with open(output_file, "w") as f:
                    json.dump(preds, f)
                with open(responses_file, "w") as f:
                    json.dump(cur_responses, f)
    except Exception:
        traceback.print_exc()
        
    with open(output_file, "w") as f:
        json.dump(preds, f)
    with open(responses_file, "w") as f:
        json.dump(cur_responses, f)
    
    print(f"Predictions saved to {output_file}")
    print(f"Responses saved to {responses_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWE Agent using RAG")
    parser.add_argument("-m", "--model", type=str, help="Model to use", default="gpt-4o-mini")
    parser.add_argument("-i", "--input_file", type=str, help="Dataset file for SWE-Bench", default="swe_bench_cache/dataset.json")
    parser.add_argument("-d", "--dir", type=str, help="Directory containing SWE-Bench task repos", default="swe_bench_cache/repos")
    parser.add_argument("-r", "--run_id", type=str, help="Id for current run", required=True)

    args = parser.parse_args()

    llm = init_backend(model=args.model)
    main(args)
        

    