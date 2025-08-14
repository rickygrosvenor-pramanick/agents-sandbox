"""
agent.py

This is the main entry point for the RAG agent. It orchestrates the entire
Retrieval-Augmented Generation pipeline by:
1.  Taking a user's prompt.
2.  Using the retriever to fetch relevant context from the vector store.
3.  Constructing an enhanced prompt that includes both the original query and
    the retrieved context.
4.  Passing the enhanced prompt to the LLM to generate a final, context-aware
    response.
"""

from retriever import query_vector_store
from utils.llm import get_response, llm, embedding_model
import os
from dotenv import load_dotenv

load_dotenv()  

def generate_enhanced_prompt(user_prompt: str, context_docs: list) -> str:
    """
    Creates an enhanced prompt for the LLM by combining the user's query
    with the retrieved context.

    Args:
        user_prompt: The original prompt from the user.
        context_docs: A list of document chunks retrieved from the vector store.

    Returns:
        A string containing the formatted, enhanced prompt.
    """
    if not context_docs:
        # If no context is found, just use the original prompt with a note.
        return f"""You are a helpful business analyst assistant.
Please answer the following user request. Note: No specific context was found in the knowledge base for this query.
User Request: {user_prompt}"""

    # Format the retrieved context
    context_str = "\n\n---\n\n".join(context_docs)

    # Construct the enhanced prompt
    prompt_template = f"""You are a helpful business analyst assistant. Your task is to answer the user's request based on the provided context.
If the context contains the necessary information, use it to formulate a detailed and accurate response.
If the context does not fully cover the user's request, use your general knowledge to supplement the answer but clearly state which parts of the answer come from the provided context.

CONTEXT:
---
{context_str}
---

USER REQUEST:
{user_prompt}
"""
    return prompt_template

def run_agent(user_prompt: str):
    """
    Runs the full RAG pipeline.

    Args:
        user_prompt: The user's question or task.
    """
    print(f"--- Running Agent for Prompt: '{user_prompt}' ---")

    # 1. Retrieve context from the vector store
    # Note: With mock embeddings, this context will be random.
    retrieved_results = query_vector_store(query=user_prompt, n_results=3)
    
    retrieved_docs = []
    if retrieved_results and retrieved_results.get('documents'):
        retrieved_docs = retrieved_results['documents'][0]
        print(f"Successfully retrieved {len(retrieved_docs)} context documents.")
    else:
        print("Warning: Could not retrieve any context from the vector store.")

    # 2. Generate the enhanced prompt
    enhanced_prompt = generate_enhanced_prompt(user_prompt, retrieved_docs)
    print("\n--- Generated Enhanced Prompt (for LLM) ---")
    print(f"{enhanced_prompt[:600]}...") # Print a snippet of the prompt

    # 3. Get the final response from the LLM
    print("\n--- Getting Final Response from LLM ---")
    final_response = get_response(user_prompt=enhanced_prompt, llm=llm)

    # 4. Print the final answer
    print("\n--- AGENT'S FINAL RESPONSE ---")
    print(final_response)


if __name__ == '__main__':
    # This is an example of how to run the agent.
    # The user's question is the starting point.
    # The agent will use this to retrieve context and generate a final answer.
    
    # Since we are using mock embeddings, the retrieved context will be random,
    # but the mock LLM will still generate its standard user story. This
    # demonstrates the end-to-end flow.
    
    prompt = "Based on the financial data, create a user story for a feature that helps financial analysts track quarterly performance."
    run_agent(prompt)
