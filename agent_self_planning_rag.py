import os
from typing import TypedDict, List, Optional
import json
from tqdm import tqdm
import argparse
import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from utils.response_utils import extract_patch_from_markdown
from prompts.self_planning_rag_prompts import *
from utils.model_utils import *
from utils.dataset_utils import *
from utils.repo_utils import *

llm = None
retriever = None

class GraphState(TypedDict):
    instance_id: str
    problem: str
    instance_repo: str
    documents: Optional[List[str]] = None
    context: Optional[str] = None
    patch: Optional[str] = None

def retrieve(state: GraphState):
    '''
    Uses vectorstore retriever to fetch top 20 code snippets using issue description as query
    '''
    global retriever
    relevant_docs = retriever.invoke(state["problem"])

    context = "\n".join([relevant_code_snippet_prompt.format(filepath = d.metadata['source'][len(state["instance_repo"]):], code = d.page_content) for d in relevant_docs])
    return {"context": context}

def generate_patch(state: GraphState):
    '''
    Generates patch condiitoned on issue description and retrieved code snippets
    '''
    inference_prompt = f"{sys_prompt}\n{problem_statement_prompt}\n{diff_patch_example}\n{final_inference_prompt}"
    prompt = ChatPromptTemplate.from_template(
        inference_prompt
    )
    
    chain = prompt | llm
    
    response = chain.invoke({
        "problem": state["problem"],
        "context": state["context"]
    })
    
    return {"patch": response.content}

def create_agent_graph():
    '''
    Workflow definition of the framework
    '''
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate_patch)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def run_inference(instance, base_repo_dir, repo_embeddings_dir, run_id):
    '''
    Initializes the agent graph and runs inference for each sample of the dataset
    '''
    global retriever
    retriever = process_repo(instance, base_repo_dir, repo_embeddings_dir)

    initial_state = GraphState(
        instance_id=instance["instance_id"],
        instance_repo=f"{base_repo_dir}/{instance['repo']}/",
        problem=instance["problem_statement"]
    )

    app = create_agent_graph()
    config = RunnableConfig(tags=[run_id])
    result = app.invoke(initial_state, config)
    patch = extract_patch_from_markdown(result["patch"])

    return result["patch"], patch

def main(args):
    run_id = args.run_id
    input_file = f"{args.dataset_dir}/dataset.json"
    repos_dir= f"{args.dataset_dir}/repos"
    repo_embeddings_dir = f"{args.dataset_dir}/repo_embeddings"

    if not os.path.exists(f"output/{run_id}"):
        os.mkdir(f"output/{run_id}")

    output_file = f"output/{run_id}/prediction_self_plan_rag_{args.model}.json"
    processed_instances = load_processed_instances(output_file)
    dataset = load_local_dataset(input_file, processed_instances)
    
    preds = [value for _, value in processed_instances.items()]
    for instance_id, instance_data in tqdm(dataset.items()):
        output = {
            "instance_id": instance_id,
            "model_name_or_path": args.model
        }
        try:
            result, patch = run_inference(instance_data, repos_dir, repo_embeddings_dir, run_id)
            output["model_patch"] = patch,
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
    parser = argparse.ArgumentParser(description="Self-Planning Agent with RAG")
    parser.add_argument("-m", "--model", type=str, help="Model to use", default="gpt-4o-mini")
    parser.add_argument("-d", "--dataset_dir", type=str, help="Directory containing SWE-Bench task repos", default="./swe_bench_verified_cache")
    parser.add_argument("-r", "--run_id", type=str, help="Id for current run", required=True)

    args = parser.parse_args()

    llm = init_backend(model=args.model)
    main(args)
        

    