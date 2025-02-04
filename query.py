#!/usr/bin/env python3
import os 
import json
import requests
import chromadb
from chromadb.config import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from ollama import Client
from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, HfApiModel, tool, GradioUI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL        = os.getenv('OLLAMA_BASE_URL')
OLLAMA_REASONING_MODEL = os.getenv('OLLAMA_REASONING_MODEL')
OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL')
OLLAMA_TOOL_MODEL      = os.getenv('OLLAMA_TOOL_MODEL')

vectordb = chromadb.PersistentClient(path="./chroma_db")
collection = vectordb.get_or_create_collection("documents")
ollama_embedding = OllamaEmbedding(
    model_name=OLLAMA_EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
    
)
reasoning_model = OpenAIServerModel(
    model_id=OLLAMA_REASONING_MODEL,
    api_base=OLLAMA_BASE_URL + "/v1",
    api_key="ollama"
)

tool_model = OpenAIServerModel(
    model_id=OLLAMA_TOOL_MODEL,
    api_base=OLLAMA_BASE_URL + "/v1",
    api_key="ollama"
)

reasoner = CodeAgent(tools=[], model=reasoning_model, add_base_tools=False, max_steps=2)

@tool
def rag_with_reasoner(user_query: str) -> str:
    """
    This is a RAG tool that takes in a user query and searches for relevant content from the vector database.
    The result of the search is given to a reasoning LLM to generate a response, so what you'll get back
    from this tool is a short answer to the user's question based on RAG context.

    Args:
        user_query: The user's question to query the vector database with.
    """
    query_emb = ollama_embedding.get_query_embedding(user_query)

    docs = collection.query(query_embeddings=[query_emb], n_results=3)
    
    context = "\n\n".join(d for d in docs.get("documents", [[]])[0])

    prompt = f"""Based on the following context, answer the user's question. Be concise and specific.
    If there isn't sufficient information, give as your answer a better query to perform RAG with.
    
Context:
{context}

Question: {user_query}

Answer:"""
    
    response = reasoner.run(prompt, reset=False)
    return response


@tool
def print_stuff(user_query: str) -> str:
    """
    This is a RAG tool allows to print anything requested by the user.

    Args:
        user_query: whatever text the user wants to print
    """
    print("FROM THE AI: " + user_query)

    return "Printed with success lord." 

primary_agent = ToolCallingAgent(tools=[rag_with_reasoner, print_stuff], model=tool_model, add_base_tools=False, max_steps=3)


def main():
    GradioUI(primary_agent).launch()

if __name__ == "__main__":
    main()