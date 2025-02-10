import os
from typing import TypedDict, List, Optional
import json

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_text_splitters import Language


class GraphState(TypedDict):
    problem: str
    documents: Optional[List[str]] = None
    context: Optional[str] = None
    patch: Optional[str] = None

def process_repo(repo_path: str):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    documents = loader.load()
    print(len(documents))


    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

def create_retriever(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(k=5)

def retrieve_documents(state: GraphState):
    retriever = create_retriever(state["documents"])
    relevant_docs = retriever.invoke(state["problem"])
    context = "\n\n".join([d.page_content for d in relevant_docs])
    return {"context": context}

def generate_patch(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        """You are a senior software engineer. Generate a code patch to resolve this issue that is part of a repository:
        
        Issue: {problem}
        
        Relevant code for the issue:
        {context}
        
        Think step by step and output the necessary changes to fix the issue.
        Format your response as a unified diff patch starting with '---' and '+++' and return the diff withing <diff/> tags.
        """
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini")
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


def run_inference(instance, repo_path):
    splits = process_repo(repo_path)

    initial_state = GraphState(
        problem=instance["problem_statement"],
        documents=splits
    )

    app = build_workflow()
    result = app.invoke(initial_state)
    
    return result["patch"]


if __name__ == "__main__":
    with open("swe_bench_cache/dataset.json", "r") as f:
        dataset = json.load(f)
    
    for key, value in dataset.items():
        test_instance = value
    
        repo_path = f"swe_bench_cache/repos/{test_instance['repo']}"
        print(repo_path)
    
        patch = run_inference(test_instance, repo_path)
        print("Generated Patch:")
        print(patch)

        break

    