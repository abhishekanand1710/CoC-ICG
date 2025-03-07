from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from typing import List, Dict, Any, Literal, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS

import git
import os
import json
import re
import argparse
from tqdm import tqdm
import traceback
import ast
from utils.repo_utils import *
from utils.response_utils import *
from utils.dataset_utils import *
from prompts_coc_v2 import *
import operator
from swe_agent_code_parser import *

llm = None
vectorstore = None
files_list = []
codebase_index = None


class GraphState(TypedDict):
    issue: str
    iterations: int
    context_chain: List[dict]
    candidate_context: Dict
    requested_context: Dict
    solution: Optional[str]
    analysis_log: Dict
    repo_path: str


def init_backend(model):
    if model in ['o3-mini', 'gpt-4o-mini']:
        return ChatOpenAI(model=model)
    else:
        raise NotImplementedError(f"Support for {model} is not yet implemented.")

def extract_filepaths(input_string):
    if input_string.startswith('[') and input_string.endswith(']'):
        items = input_string[1:-1].split(',')
        return [item.strip().strip('"\'') for item in items]
    return [input_string]

def get_repository_files_list(base_path, repo_path: str) -> str:
    target_path = os.path.join(base_path, repo_path)
    
    file_paths = []
    for root, _, files in os.walk(target_path):
        for file in files:
            path = os.path.join(root, file).removeprefix(base_path)
            if 'tests/' in path:
                continue
            file_paths.append(path)
    # with open('file.txt', 'w') as f:
    #     f.write('\n'.join(file_paths))
    return file_paths

def get_repository_files(repo_path: str, path: str = "") -> str:
    """List files at a specific path in the repository."""
    target_path = os.path.join(repo_path, path)
    
    if not os.path.exists(target_path):
        return f"Path '{path}' does not exist"
    
    result = []
    for root, dirs, files in os.walk(target_path):
        rel_root = os.path.relpath(root, repo_path)
        if rel_root == ".":
            rel_root = ""
            
        for d in dirs:
            rel_path = os.path.join(rel_root, d)
        
        for f in files:
            rel_path = os.path.join(rel_root, f)
            result.append(f"📄 {rel_path}")
            
        break
    
    return "\n".join(result)

def format_context_chain(context_chain: List[Dict]):
    context_str = ""
    for context in context_chain:
        content = context['content'].removeprefix('```python')
        content = content.removesuffix('```')

        name = context["name"]
        if not name.endswith('.py'):
            name = name.split(".")[-1]

        context = CONTEXT_TEMPLATE.format(
            name = name,
            file_path = context["file_path"],
            content = content,
            iteration = context["retrieved_at"]
        )
        context_str += context

    return context_str

#####################################################


## stages - nodes

def initialize(state: GraphState):
    return {
        "issue": state["issue"],
        "iterations": 0,
        "context_chain": [],
        "candidate_context": [],
        "requested_context": [],
        "solution": None,
        "analysis_log": {},
        "repo_path": state["repo_path"]
    }

def analyze(state: GraphState):
    context_str = format_context_chain(state["context_chain"])
    if not context_str:
        context_str = "No context fetched yet."

    agent_prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
    chain = agent_prompt | llm
    result = chain.invoke({
        "issue_description": state["issue"],
        "cur_iteration": state["iterations"],
        "file_structure": "File structure not available",
        "context_str": context_str,
        "analysis_log": '\n'.join(state["analysis_log"].values())
    })
    response = result.content
    context = parse_analysis_response(response)
    
    if context:
        return {
            "iterations": state["iterations"] + 1,
            "requested_context": context
        }
    else:
        return {
            "iterations": state["iterations"] + 1,
            "solution": context
        }

def retrieve(state: GraphState):
    retrieved = {}
    requested_context = state["requested_context"]
    functions = requested_context["functions"]
    classes = requested_context["classes"]
    files = requested_context["files"]
    others = requested_context["others"]

    analysis_log = state["analysis_log"]

    for func in functions:
        fetched_context = query_context(func, "function", codebase_index, state["repo_path"])
        if not fetched_context:
            continue
        name, data = fetched_context
        retrieved[func] = {
            "name": name,
            "file_path": data["file_path"],
            "content": data["definition"],
            "retrieved_at": state["iterations"]
        }
        analysis_log[func] = f"NEED_CONTEXT: FUNCTION = {func}"

    for cls_ in classes:
        fetched_context = query_context(cls_, "class", codebase_index, state["repo_path"])
        if not fetched_context:
            continue
        name, data = fetched_context
        retrieved[cls_] = {
            "name": name,
            "file_path": data["file_path"],
            "content": data["definition"],
            "retrieved_at": state["iterations"]
        }
        analysis_log[cls_] = f"NEED_CONTEXT: CLASS = {cls_}"

    for file in files:
        fetched_context = query_context(file, "file", codebase_index, state["repo_path"])
        if not fetched_context:
            continue
        file_path, data = fetched_context
        retrieved[file] = {
            "name": file_path.split("/")[-1],
            "file_path": file_path,
            "content": data,
            "retrieved_at": state["iterations"]
        }
        analysis_log[file] = f"NEED_CONTEXT: FILE = {file}"

    for entity, file in others.items():
        fetched_context = query_context(file, "file", codebase_index, state["repo_path"])
        if not fetched_context:
            continue
        file_path, data = fetched_context
        retrieved[entity] = {
            "name": file_path.split("/")[-1],
            "file_path": file_path,
            "content": data,
            "retrieved_at": state["iterations"]
        }
        analysis_log[file] = f"NEED_CONTEXT: OTHER = {entity}@{file}"
    
    return {
        "context_chain": state["context_chain"],
        "candidate_context": retrieved,
        "requested_context": [],
        "analysis_log": analysis_log
    }

def filter_context(state: GraphState):
    context_chain = state["context_chain"]
    candidate_context = state["candidate_context"]

    analysis_log = state["analysis_log"]

    if candidate_context:
        context_str = format_context_chain(state["context_chain"])
        for query, context in candidate_context.items():
            candidate_context_str = format_context_chain([context])

            agent_prompt = ChatPromptTemplate.from_template(FILTER_PROMPT)
            chain = agent_prompt | llm
            result = chain.invoke({
                "issue_description": state["issue"],
                "context_str": context_str,
                "candidate_context_str": candidate_context_str,
            })
            response = result.content

            is_relevant, code = parse_filter_response(response)
            if is_relevant:
                if code:
                    context["content"] = code
                context_chain.append(context)
                analysis_log[query] = analysis_log[query] + " -> Code was relevant and is included above."
            else:
                if query in analysis_log:
                    analysis_log[query] = analysis_log[query] + " -> Code was irrelevant. Don't request it again."

        
    return {
        "context_chain": state["context_chain"],
        "analysis_log": analysis_log
    }

def create_agent_graph():
    workflow = StateGraph(GraphState)
 
    workflow.add_node("init", initialize)
    workflow.add_node("analyze", analyze)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("filter", filter_context)
    workflow.add_node("solve", lambda s: s)

    workflow.add_conditional_edges(
        "analyze",
        lambda s: "retrieve" if len(s["requested_context"]) > 0 else "solve",
        {"retrieve": "retrieve", "solve": "solve"}
    )

    workflow.add_edge("init", "analyze")
    workflow.add_edge("retrieve", "filter")
    workflow.add_edge("filter", "analyze")
    workflow.add_edge("solve", END)

    workflow.set_entry_point("init")
    
    return workflow.compile()

def run_inference(instance, base_repo_dir, max_steps = 10):
    global files_list, vectorstore, codebase_index
    repo_path = process_repo_min(instance, base_repo_dir)

    codebase_index = analyze_codebase(repo_path)
    
    app = create_agent_graph()
    files_list = get_repository_files_list(base_repo_dir, instance['repo'])
    
    result = app.invoke(GraphState(issue=instance["problem_statement"], repo_path=repo_path))
    return result

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
        cnt = 0
        for key, test_instance in tqdm(dataset.items()):
            # if cnt == 0:
            #     cnt += 1
            #     continue
            result = run_inference(test_instance, args.dir)
            print(result["final_patch"])
            break
            # output = {
            #     "instance_id": key,
            #     "model_patch": patch,
            #     "model_name_or_path": args.model
            # }
            # preds.append(output)

            # response_output = {
            #     "instance_id": key,
            #     "response": result,
            #     "model_name_or_path": args.model
            # }
            # cur_responses.append(response_output)

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
    parser.add_argument("-i", "--input_file", type=str, help="Dataset file for SWE-Bench", default="./swe_bench_cache/dataset.json")
    parser.add_argument("-d", "--dir", type=str, help="Directory containing SWE-Bench task repos", default="./swe_bench_cache/repos")
    parser.add_argument("-r", "--run_id", type=str, help="Id for current run", required=True)

    args = parser.parse_args()

    llm = init_backend(model=args.model)
    main(args)
        