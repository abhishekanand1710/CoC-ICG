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
from prompts_coc import *

llm = None
vectorstore = None
files_list = []

class GraphState(TypedDict):
    problem: str
    repo_path: str
    steps_history: List[Dict[str, Any]]
    current_step_number: int
    context: Dict[str, Any]
    max_steps: int
    final_patch: Optional[str]
    instance_repo: str


def init_backend(model):
    if model in ['gpt-4o-mini']:
        return ChatOpenAI(model=model)
    else:
        raise NotImplementedError(f"Support for {model} is not yet implemented.")

def retrieve_documents(state: GraphState, query: str):
    test_files = [f"swe_bench_cache/repos{d}" for d in files_list if 'tests/' in d]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10, "filter": {"source": {"$nin": test_files}}})
    relevant_docs = retriever.invoke(query)
    

    context = "\n".join([relevant_code_snippet_prompt.format(filepath = d.metadata['source'], code = d.page_content) for d in relevant_docs])
    return {"context": context}

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
            file_paths.append(os.path.join(root, file).removeprefix(base_path))
    with open('file.txt', 'w') as f:
        f.write('\n'.join(file_paths))
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

def get_file_content(file_path: str) -> str:
    """Get the content of a specific file in the repository."""

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

def search_repository(repo_path: str, query: str) -> str:
    """Search for files in the repository that match the query."""
    matches = []
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if '.git' in root:
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)
            
            if query.lower() in file.lower():
                matches.append(f"File: {rel_path} (Match in filename)")
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if query.lower() in content.lower():
                        matches.append(f"File: {rel_path} (Match in content)")
            except:
                pass
    
    return json.dumps(matches[:10])

def execute_agent_step(state: GraphState):
    """Run the agent to determine the next step or action."""
    steps_history_formatted = json.dumps(state["steps_history"], indent=2)
    context_formatted = json.dumps(state["context"], indent=2)

    agent_prompt = ChatPromptTemplate.from_template(coc_prompt)
    
    chain = agent_prompt | llm
    result_ = chain.invoke({
        "problem": state["problem"],
        "repo_path": state["repo_path"],
        "current_step": state["current_step_number"],
        "steps_history": steps_history_formatted,
        "context": context_formatted,
        "max_steps": state["max_steps"]
    })
    result = result_.content
    
    action_match = re.search(r'ACTION: ([A-Z_]+)', result)
    
    if not action_match:
        action = "REQUEST_CONTEXT"
        reason = "Missing action specification"
        request = "Please provide basic repository structure"
    else:
        action = action_match.group(1)
    
    step_record = {
        "step_number": state["current_step_number"],
        # "agent_response": result,
        "action": action
    }
    
    if action == "REQUEST_CONTEXT":
        reason_match = re.search(r'REASON: (.*?)$', result, re.MULTILINE)
        request_match = re.search(r'REQUEST: (.*?)$', result, re.MULTILINE | re.DOTALL)
        
        if reason_match and request_match:
            reason = reason_match.group(1).strip()
            request = request_match.group(1).strip()
            
            step_record["reason"] = reason
            step_record["request"] = request
        
    elif action == "IMPLEMENT":
        description_match = re.search(r'DESCRIPTION: (.*?)$', result, re.MULTILINE)
        code_changes_match = re.search(r'CODE_CHANGES:\s*\[(.*?)\]', result, re.MULTILINE | re.DOTALL)
        
        if description_match:
            description = description_match.group(1).strip()
            step_record["description"] = description
            
            if code_changes_match:
                try:
                    code_changes_text = "[" + code_changes_match.group(1) + "]"
                    code_changes = json.loads(code_changes_text)
                    step_record["code_changes"] = code_changes
                except:
                    step_record["code_changes"] = {"error": "Failed to parse code changes"}
            
            state["current_step_number"] += 1
            
    elif action == "FINALIZE":
        pass

    state["steps_history"].append(step_record)
    
    return state

def gather_context(state: GraphState):
    last_step = state["steps_history"][-1]
    
    if last_step["action"] != "REQUEST_CONTEXT":
        return state
    
    request = last_step.get("request", "")
    reason = last_step.get("reason", "")
    
    file_paths = extract_filepaths(request)
    
    gathered_context = {}

    print(reason)
    content = retrieve_documents(state, reason)
    gathered_context[f"files_{', '.join(file_paths)}"] = content
    
    if not gathered_context:
        if "auth" in request.lower() or "login" in request.lower() or "password" in request.lower():
            try:
                results = search_repository(state["repo_path"], "auth")
                gathered_context["search_auth"] = results
            except Exception:
                pass
    
    for key, value in gathered_context.items():
        state["context"][key] = value
    
    return state

def should_continue(state: GraphState) -> Literal["continue", "stop"]:
    """Determine if the workflow should continue or stop."""
    if state["current_step_number"] >= state["max_steps"]:
        return "stop"
    if state["final_patch"] is not None:
        return "stop"
    return "continue"

def create_agent_graph():
    workflow = StateGraph(GraphState)
 
    workflow.add_node("agent_step", execute_agent_step)
    workflow.add_node("gather_context", gather_context)

    workflow.set_entry_point("agent_step")
    workflow.add_edge("agent_step", "gather_context")
    workflow.add_conditional_edges(
        "gather_context",
        should_continue,
        {
            "continue": "agent_step",
            "stop": END
        }
    )
    
    return workflow.compile()

def run_inference(instance, base_repo_dir, max_steps = 10):
    global files_list, vectorstore
    vectorstore = process_repo(instance, base_repo_dir)
    
    app = create_agent_graph()
    files_list = get_repository_files_list(base_repo_dir, instance['repo'])
    
    initial_state = GraphState(
        instance_id=instance["instance_id"],
        problem=instance["problem_statement"],
        repo_path=f"{base_repo_dir}/{instance['repo']}/",
        steps_history=[],
        current_step_number=1,
        context={},
        messages=[],
        max_steps=max_steps,
        final_patch=None,
        instance_repo=f"{base_repo_dir}/{instance['repo']}/",
    )
    
    result = app.invoke(initial_state)
    return result


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

    llm.bind_tools(tools)
    
    preds = [value for _, value in processed_instances.items()]
    cur_responses = [value for _, value in responses.items()]
    try:
        for key, test_instance in tqdm(dataset.items()):
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
        