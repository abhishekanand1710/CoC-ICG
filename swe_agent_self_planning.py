from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from typing import List, Dict, Any, Literal, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.config import RunnableConfig

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
from prompts_self_planning import *
import operator
from swe_agent_code_parser import *

llm = None
llm_greedy = None
vectorstore = None
files_list = []
codebase_index = None

MAX_STEPS = 70

class GraphState(TypedDict):
    instance_id: str
    issue: str
    context_chain: List[dict]
    retrieved_keys: set
    candidate_context: Dict
    requested_context: Dict
    analysis: Optional[str]
    solution: Optional[str]
    analysis_log: Dict
    test_cases: List[str]
    repo_path: str
    patch_files: List[str]
    file_edits: Dict
    repo_structure: str


def init_backend(model):
    if model in ['o3-mini', 'gpt-4o-mini', 'gpt-4o']:
        return ChatOpenAI(model=model)
    else:
        raise NotImplementedError(f"Support for {model} is not yet implemented.")
    
def init_backend_greedy(model):
    if model in ['o3-mini', 'gpt-4o-mini', 'gpt-4o']:
        if model == 'o3-mini':
            return ChatOpenAI(model=model)
        return ChatOpenAI(model=model, temperature=0.1, top_p=0.1)
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
        content = content.removesuffix('```').strip()

        name = context["name"]
        if not name.endswith('.py'):
            name = name.split(".")[-1]

        # if "additional_content" in context:
        #     context = CONTEXT_TEMPLATE_EXTN.format(
        #         name = name,
        #         file_path = context["file_path"],
        #         content = content,
        #         iteration = context["retrieved_at"],
        #         additional_content = context["additional_content"]
        #     )
        #     context_str += context
        # else:
        context = CONTEXT_TEMPLATE.format(
            name = name,
            file_path = context["file_path"],
            content = content
        )
        context_str += context

    return context_str

def format_candidate_context(context: Dict, reason: str):
    context_str = ""
    content = context['content'].removeprefix('```python')
    content = content.removesuffix('```').strip()

    name = context["name"]
    if not name.endswith('.py'):
        name = name.split(".")[-1]

    context_str = CANDIDATE_CONTEXT_TEMPLATE.format(
        name = name,
        reason = reason,
        file_path = context["file_path"],
        content = content,
        iteration = context["retrieved_at"]
    )

    return context_str

#####################################################


## stages - nodes

def initialize(state: GraphState):
    return {
        "instance_id": state["instance_id"],
        "issue": state["issue"],
        "context_chain": [],
        "retrieved_keys": set(),
        "candidate_context": [],
        "requested_context": [],
        "analysis": None,
        "solution": None,
        "test_cases": [],
        "repo_path": state["repo_path"],
        "patch_files": List[str],
        "file_edits": {}
    }

def analyze(state: GraphState):
    context_str = format_context_chain(state["context_chain"])
    if not context_str:
        context_str = "No context fetched yet."

    agent_prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
    chain = agent_prompt | llm

    result = chain.invoke({
        "issue_description": state["issue"],
        "modules": '\n'.join(codebase_index['modules'].keys()),
        "context_str": context_str
    })
    
    response = result.content
    _, parsed_response = parse_analysis_response(response)

    return {
        "requested_context": parsed_response,
        "analysis": response
    }

def retrieve_cumulative(state: GraphState):
    retrieved = {}
    requested_context = state["requested_context"]
    functions = requested_context["functions"]
    classes = requested_context["classes"]
    files = requested_context["files"]
    modules = requested_context["modules"]
    others = requested_context["others"]

    queries = []
    reasons = {}
    for f in functions: 
        queries.append({'query': f[0], 'type': 'function'})
        reasons[f[0]] = f[1]
    for c in classes: 
        queries.append({'query': c[0], 'type': 'class'})
        reasons[c[0]] = c[1]
    for f in files: 
        queries.append({'query': f[0], 'type': 'file'})
        reasons[f[0]] = f[1]
    for m in modules: 
        queries.append({'query': m[0], 'type': 'module'})
        reasons[m[0]] = m[1]
    for key, o in others.items(): 
        query = f'{key}@{o[0]}'
        queries.append({'query': query, 'type': 'other'})
        reasons[query] = o[1]

    retrieved_data = query_cumulative(queries, codebase_index, state["repo_path"])

    retrieved_keys = state["retrieved_keys"]

    for func in functions:
        key, reason = func
        if key in retrieved_keys:
            if key in retrieved_data:
                del retrieved_data[key]
            continue

        retrieved_keys.add(key)
        if key not in retrieved_data:
            continue
        name, data = retrieved_data[key]
        retrieved[key] = {
            "name": name,
            "file_path": data["file_path"],
            "content": data["definition"],
            "reason": reason
        }
    for cls_ in classes:
        key, reason = cls_
        if key in retrieved_keys:
            if key in retrieved_data:
                del retrieved_data[key]
            continue

        retrieved_keys.add(key)
        if key not in retrieved_data:
            continue
        name, data = retrieved_data[key]
        retrieved[key] = {
            "name": name,
            "file_path": data["file_path"],
            "content": data["definition"],
            "reason": reason
        }

    for file in files:
        key, reason = file
        if key in retrieved_keys:
            if key in retrieved_data:
                del retrieved_data[key]
            continue

        retrieved_keys.add(key)
        if key not in retrieved_data:
            continue
        file_path, data = retrieved_data[key]
        # additional_content = get_classes_functions_for_file(file_path, codebase_index)
        retrieved[key] = {
            "name": file_path.split("/")[-1],
            "file_path": file_path,
            "content": data,
            "reason": reason,
            # "additional_content": additional_content
        }

    for key, reason in modules:
        if key in retrieved_keys:
            if key in retrieved_data:
                del retrieved_data[key]
            continue

        retrieved_keys.add(key)
        if key not in retrieved_data:
            continue
        file_path, data = retrieved_data[key]
        # additional_content = get_classes_functions_for_file(file_path, codebase_index)
        retrieved[key] = {
            "name": key,
            "file_path": file_path,
            "content": data,
            "reason": reason,
            # "additional_content": additional_content
        }

    for entity, file in others.items():
        key, reason = file
        if key in retrieved_keys:
            if key in retrieved_data:
                del retrieved_data[key]
            continue

        retrieved_keys.add(key)
        if key not in retrieved_data:
            continue
        file_path, data = retrieved_data[key]
        # additional_content = get_classes_functions_for_file(file_path, codebase_index)
        retrieved[f"{entity}@{key}"] = {
            "name": f"{entity}@{key}",
            "file_path": file_path,
            "content": data,
            "reason": reason,
            # "additional_content": additional_content
        }
    
    return {
        "context_chain": retrieved.values(),
        "requested_context": [],
        
        "retrieved_keys": retrieved_keys
    }

def retrieve(state: GraphState):
    retrieved = {}
    requested_context = state["requested_context"]
    functions = requested_context["functions"]
    classes = requested_context["classes"]
    files = requested_context["files"]
    modules = requested_context["modules"]
    others = requested_context["others"]

    analysis_log = state["analysis_log"]

    retrieved_keys = state["retrieved_keys"]

    for func in functions:
        key, reason = func
        if key in retrieved_keys:
            continue
        retrieved_keys.add(key)

        fetched_context = query_context(key, "function", codebase_index, state["repo_path"])
        if not fetched_context:
            continue
        name, data = fetched_context
        retrieved[key] = {
            "name": name,
            "file_path": data["file_path"],
            "content": data["definition"],
            "reason": reason
        }
        analysis_log[key] = f"NEED_CONTEXT: FUNCTION = {key}"

    for cls_ in classes:
        key, reason = cls_
        if key in retrieved_keys:
            continue
        retrieved_keys.add(key)

        fetched_context = query_context(key, "class", codebase_index, state["repo_path"])
        if not fetched_context:
            continue
        name, data = fetched_context
        retrieved[key] = {
            "name": name,
            "file_path": data["file_path"],
            "content": data["definition"],
            "reason": reason
        }
        analysis_log[key] = f"NEED_CONTEXT: CLASS = {key}"

    for file in files:
        key, reason = file
        if key in retrieved_keys:
            continue
        retrieved_keys.add(key)

        fetched_context = query_context(key, "file", codebase_index, state["repo_path"])
        if not fetched_context:
            continue
        file_path, data = fetched_context
        retrieved[key] = {
            "name": file_path.split("/")[-1],
            "file_path": file_path,
            "content": data,
            "reason": reason
        }
        analysis_log[key] = f"NEED_CONTEXT: FILE = {key}"

    for key, reason in modules:
        if key in retrieved_keys:
            continue
        retrieved_keys.add(key)

        fetched_context = query_context(key, "module", codebase_index, state["repo_path"])
        if not fetched_context:
            continue
        file_path, data = fetched_context
        retrieved[key] = {
            "name": key,
            "file_path": file_path,
            "content": data,
            "reason": reason
        }
        analysis_log[key] = f"NEED_MODULE: {key}"

    for entity, file in others.items():
        key, reason = file
        if key in retrieved_keys:
            continue
        retrieved_keys.add(key)

        fetched_context = query_context(key, "file", codebase_index, state["repo_path"])
        if not fetched_context:
            continue
        file_path, data = fetched_context
        retrieved[f"{entity}@{key}"] = {
            "name": f"{entity}@{key}",
            "file_path": file_path,
            "content": data,
            "reason": reason
        }
        analysis_log[f"{entity}@{key}"] = f"NEED_CONTEXT: OTHER = {entity}@{key}"
    
    return {
        "context_chain": retrieved.values(),
        "requested_context": [],
        "analysis_log": analysis_log,
        "retrieved_keys": retrieved_keys
    }

def solve(state: GraphState):
    context_str = format_context_chain(state["context_chain"])
    agent_prompt = ChatPromptTemplate.from_template(SOLVE_AND_ANALYZE_PROMPT)
    chain = agent_prompt | llm_greedy
    result = chain.invoke({
        "issue_description": state["issue"],
        "context_str": context_str
    })
    response = result.content
    return {
        "solution": response
    }

def localize(state: GraphState):
    solution = state["solution"]
    context_str = format_context_chain(state["context_chain"])
    agent_prompt = ChatPromptTemplate.from_template(REQUEST_FILES_TO_BE_EDITED_PROMPT)
    chain = agent_prompt | llm
    result = chain.invoke({
        "issue_description": state["issue"],
        "code_context": context_str,
        "solution": solution
    })
    response = result.content
    patch_files = parse_file_request_response(response)
    return {
        "patch_files": patch_files
    }

def generate_patch(state: GraphState):
    patch_files = state["patch_files"]
    context_chain = state["context_chain"]

    patch_context_dict = {file: [] for file in patch_files}
    for context in context_chain:
        if context["file_path"] in patch_context_dict:
            patch_context_dict[context["file_path"]].append(context)

    edited_patch_files = []
    file_edit_snippets_dict = {}
    for file, file_relevant_context in patch_context_dict.items():
        file_relevant_context_str = format_context_chain(file_relevant_context)

        agent_prompt = ChatPromptTemplate.from_template(PATCH_GENERATION_PROMPT)
        chain = agent_prompt | llm
        result = chain.invoke({
            "issue_description": state["issue"],
            "solution": state["solution"],
            "patch_files": "\n".join(patch_files),
            "edited_patch_files": "\n".join(edited_patch_files),
            "current_file": file,
            "cur_code_context": file_relevant_context_str
        })
        response = result.content
        file_edit_snippets = parse_file_edit_response(response)
        file_edit_snippets_dict[file] = file_edit_snippets

    return {
        "file_edits": file_edit_snippets_dict
    }

def create_agent_graph():
    workflow = StateGraph(GraphState)
 
    workflow.add_node("init", initialize)
    workflow.add_node("analyze", analyze)
    workflow.add_node("retrieve", retrieve_cumulative)
    workflow.add_node("solve", solve)
    workflow.add_node("localize", localize)
    workflow.add_node("generate_patch", generate_patch)
    workflow.add_node("save", lambda s: s)

    workflow.add_edge("init", "analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "solve")
    workflow.add_edge("solve", "localize")
    workflow.add_edge("localize", "generate_patch")
    workflow.add_edge("generate_patch", "save")
    workflow.add_edge("save", END)

    workflow.set_entry_point("init")
    
    return workflow.compile()

def run_inference(instance, base_repo_dir, run_id):
    global files_list, vectorstore, codebase_index

    repo_path = process_repo_min(instance, base_repo_dir)
    repo_structure = generate_repo_structure(repo_path)
    codebase_index = analyze_codebase(repo_path)
    
    app = create_agent_graph()
    config = RunnableConfig(recursion_limit=MAX_STEPS, tags=[run_id])

    files_list = get_repository_files_list(base_repo_dir, instance['repo'])
    
    result = app.invoke(
        GraphState(
            instance_id=instance["instance_id"],
            issue=instance["problem_statement"], 
            repo_path=repo_path,
            repo_structure=repo_structure
        ), 
        config=config
    )
    return result

def main(args):
    run_id = args.run_id
    if not os.path.exists(f"output/{run_id}"):
        os.mkdir(f"output/{run_id}")

    output_file = f"output/{run_id}/prediction_coc_{args.model}.json"
    processed_instances = load_processed_instances(output_file)

    dataset = load_local_dataset(args.input_file, processed_instances)
    print(f"Generating for {len(dataset)} samples.")
    
    preds = [value for _, value in processed_instances.items()]
    try:
        for key, test_instance in tqdm(dataset.items()):
            try:
                result = run_inference(test_instance, args.dir, run_id)
                result_patch = generate_patch_file(result["file_edits"], result["repo_path"])
                output = {
                    "instance_id": key,
                    "model_patch": result_patch,
                    "model_name_or_path": args.model
                }
                # print(result_patch)
                # break
                preds.append(output)
            except Exception:
                traceback.print_exc()
                output = {
                    "instance_id": key,
                    "model_patch": "",
                    "model_name_or_path": args.model
                }
                preds.append(output)

            if len(preds)%5 == 0:
                with open(output_file, "w") as f:
                    json.dump(preds, f)
            
    except Exception:
        traceback.print_exc()
        
    with open(output_file, "w") as f:
        json.dump(preds, f)
    # with open(responses_file, "w") as f:
    #     json.dump(cur_responses, f)
    
    print(f"Predictions saved to {output_file}")
    # print(f"Responses saved to {responses_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWE Agent using RAG")
    parser.add_argument("-m", "--model", type=str, help="Model to use", default="gpt-4o-mini")
    parser.add_argument("-i", "--input_file", type=str, help="Dataset file for SWE-Bench", default="./swe_bench_verified_cache/dataset.json")
    parser.add_argument("-d", "--dir", type=str, help="Directory containing SWE-Bench task repos", default="./swe_bench_verified_cache/repos")
    parser.add_argument("-r", "--run_id", type=str, help="Id for current run", required=True)

    args = parser.parse_args()

    llm = init_backend(model=args.model)
    llm_greedy = init_backend(model=args.model)
    main(args)
        