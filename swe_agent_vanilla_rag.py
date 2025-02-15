import os
from typing import TypedDict, List, Optional
import json
from tqdm import tqdm
import argparse

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_text_splitters import Language

from utils.git_utils import checkout_git_repo_at_commit
from utils.response_utils import extract_diff_from_markdown
from prompts import *

llm = None

class GraphState(TypedDict):
    instance_id: str
    problem: str
    documents: Optional[List[str]] = None
    context: Optional[str] = None
    patch: Optional[str] = None

def init_backend(model):
    if model in ['gpt-4o-mini']:
        return ChatOpenAI(model=model)
    else:
        raise NotImplementedError(f"Support for {model} is not yet implemented.")

def process_repo(instance, base_dir: str):
    repo_dir = checkout_git_repo_at_commit(base_dir, repo_name=instance['repo'], commit=instance['base_commit'])

    loader = GenericLoader.from_filesystem(
        repo_dir,
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
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

    return vectorstore.as_retriever(k=20)

def retrieve_documents(state: GraphState):
    retriever = create_retriever(state["instance_id"], state["documents"])
    relevant_docs = retriever.invoke(state["problem"])
    context = "\n\n".join([d.page_content for d in relevant_docs])
    return {"context": context}

def generate_patch(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        f"{sys_prompt}\n{problem_staement_prompt}\n{unified_diff_prompt}"
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
        problem=instance["problem_statement"],
        documents=splits
    )

    app = build_workflow()
    result = app.invoke(initial_state)
    patch = extract_diff_from_markdown(result["patch"])

    return patch

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
    output_file = f"output/prediction_vanilla_rag_{args.model}.json"
    processed_instances = load_processed_instances(output_file)

    dataset = load_local_dataset(args.input_file, processed_instances)
    print(f"Generating for {len(dataset)} samples.")
    
    preds = [value for _, value in processed_instances.items()]
    try:
        for key, test_instance in tqdm(dataset.items()):
            patch = run_inference(test_instance, args.repo_dir)
            output = {
                "instance_id": key,
                "model_patch": patch,
                "model_name_or_path": args.model
            }
            preds.append(output)
            if len(preds) == 50:
                break
    except Exception as e:
        print(e)

    with open(output_file, "w") as f:
        json.dump(preds, f)
    
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWE Agent using RAG")
    parser.add_argument("-m", "--model", type=str, help="Model to use", default="gpt-4o-mini")
    parser.add_argument("-i", "--input_file", type=str, help="Dataset file for SWE-Bench", default="swe_bench_cache/dataset.json")
    parser.add_argument("-r", "--repo_dir", type=str, help="Directory containing SWE-Bench task repos", default="swe_bench_cache/repos")

    args = parser.parse_args()

    llm = init_backend(model=args.model)
    main(args)
        

    