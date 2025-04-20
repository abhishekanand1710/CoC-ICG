import os
import json
import argparse
from tqdm import tqdm
import traceback
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.runnables.config import RunnableConfig
from utils.repo_utils import *
from utils.response_utils import *
from utils.dataset_utils import *
from prompts.self_planning_icr_prompts import *
from code_indexer import *
from utils.prompt_utils import *
from utils.model_utils import *

llm = None
llm_greedy = None
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
    '''
    Analyze stage - Takes in issue description, all previous sub-task descriptions, and all relevant retrieved code context as input
    and outputs code context requests
    '''
    context_str = format_context_chain_icr(state["context_chain"])
    if not context_str:
        context_str = "No context fetched yet."

    agent_prompt = ChatPromptTemplate.from_template(ANALYSIS_ICR_PROMPT)
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

def retrieve(state: GraphState):
    '''
    Retrieve stage - Takes in all the context requests for files, functions, classes, modules and other entities 
    from analyze stage subtask description as input and retrieves corresponding code snippets from the repository's
    codebase index
    '''
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
        retrieved[key] = {
            "name": file_path.split("/")[-1],
            "file_path": file_path,
            "content": data,
            "reason": reason,
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
        retrieved[key] = {
            "name": key,
            "file_path": file_path,
            "content": data,
            "reason": reason,
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
        retrieved[f"{entity}@{key}"] = {
            "name": f"{entity}@{key}",
            "file_path": file_path,
            "content": data,
            "reason": reason,
        }
    
    return {
        "context_chain": retrieved.values(),
        "requested_context": [],
        
        "retrieved_keys": retrieved_keys
    }

def solve(state: GraphState):
    '''
    Solve stage - Takes in the issue description, root cause analysis generated as the last sub-task by analyze stage, and all relevant context
    and generates the solution as a diff patch.
    '''
    context_str = format_context_chain_icr(state["context_chain"])
    agent_prompt = ChatPromptTemplate.from_template(SOLVE_PROMPT)
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
    '''
    Localize stage - Takes in the generated solution and outputs a list of files to edit
    '''
    solution = state["solution"]
    context_str = format_context_chain_icr(state["context_chain"])
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
    '''
    Edit stage - Takes in the requested files and generated solution as input and outputs edit commands to be applied to the files.
    '''
    patch_files = state["patch_files"]
    context_chain = state["context_chain"]

    patch_context_dict = {file: [] for file in patch_files}
    for context in context_chain:
        if context["file_path"] in patch_context_dict:
            patch_context_dict[context["file_path"]].append(context)

    edited_patch_files = []
    file_edit_snippets_dict = {}
    for file, file_relevant_context in patch_context_dict.items():
        file_relevant_context_str = format_context_chain_icr(file_relevant_context)

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
    '''
    Workflow definition of the framework
    '''
    workflow = StateGraph(GraphState)
 
    workflow.add_node("init", initialize)
    workflow.add_node("analyze", analyze)
    workflow.add_node("retrieve", retrieve)
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
    '''
    Initializes the agent graph and runs inference for each sample of the dataset
    '''
    global codebase_index

    repo_path = process_repo_min(instance, base_repo_dir)
    repo_structure = generate_repo_structure(repo_path)
    codebase_index = analyze_codebase(repo_path)
    
    app = create_agent_graph()
    config = RunnableConfig(recursion_limit=MAX_STEPS, tags=[run_id])

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
    input_file = f"{args.dataset_dir}/dataset.json"
    repos_dir= f"{args.dataset_dir}/repos"

    if not os.path.exists(f"output/{run_id}"):
        os.mkdir(f"output/{run_id}")

    output_file = f"output/{run_id}/prediction_self_plan_icr_{args.model}.json"
    processed_instances = load_processed_instances(output_file)
    dataset = load_local_dataset(input_file, processed_instances)
    
    preds = [value for _, value in processed_instances.items()]
    for instance_id, instance_data in tqdm(dataset.items()):
        output = {
            "instance_id": instance_id,
            "model_name_or_path": args.model
        }
        try:
            result = run_inference(instance_data, repos_dir, run_id)
            result_patch = generate_patch_file(result["file_edits"], result["repo_path"])
            output["model_patch"] = result_patch,
        except Exception:
            traceback.print_exc()
            output["model_patch"] = ""
        
        preds.append(output)

        if len(preds)%5 == 0:
            with open(output_file, "w") as f:
                json.dump(preds, f)
            
    with open(output_file, "w") as f:
        json.dump(preds, f)
    
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Planning Agent with Indexed Context Retrieval")
    parser.add_argument("-m", "--model", type=str, help="Model to use", default="gpt-4o-mini")
    parser.add_argument("-d", "--dataset_dir", type=str, help="Directory containing SWE-Bench task repos", default="./swe_bench_verified_cache")
    parser.add_argument("-r", "--run_id", type=str, help="Id for current run", required=True)

    args = parser.parse_args()

    llm = init_backend(model=args.model)
    llm_greedy = init_backend(model=args.model)
    main(args)
        