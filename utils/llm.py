"""
llm.py

A utility module for interacting with a LangChain LLM.
This provides a simple function to get a response from a pre-initialized model.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()  

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=500,
    streaming=False,
    api_key=os.getenv("OPENAI_API_KEY")
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_response(user_prompt: str, llm: ChatOpenAI = llm) -> str:
    """
    Gets a response from the provided LangChain ChatOpenAI model.

    Args:
        user_prompt: The user's input prompt as a string.
        llm: An initialized instance of ChatOpenAI.

    Returns:
        The response from the LLM.
    """
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("The 'llm' parameter must be an instance of ChatOpenAI.")
    
    messages = [HumanMessage(content=user_prompt)]
    
    try:
        result: LLMResult = llm.generate([messages])
        response_content = result.generations[0][0].text
        return response_content
    except Exception as e:
        print(f"An error occurred while communicating with the LLM: {e}")
        return "Sorry, I was unable to get a response from the model."


def embed_text(text: str, embedding_model: OpenAIEmbeddings = embedding_model) -> List[float]:
    """
    Generates an embedding for the given text using the provided model.

    Args:
        text: The text to embed.
        embedding_model: An initialized instance of OpenAIEmbeddings.

    Returns:
        An embedding vector (list of floats).
    """
    if not isinstance(embedding_model, OpenAIEmbeddings):
        raise TypeError("The 'embedding_model' parameter must be an instance of OpenAIEmbeddings.")
    
    try:
        return embedding_model.embed_query(text)
    except Exception as e:
        print(f"An error occurred while creating the embedding: {e}")
        return []


if __name__ == '__main__':
    # This is an example of how to use the get_response function with a real API key.
    # Make sure your .env file has your OPENAI_API_KEY.
    
    # 1. The models are already initialized at the top of the file.
    
    # 2. Define a prompt
    prompt = "What is the capital of France?"

    # 3. Get the response
    print(f"User Prompt: {prompt}")
    response = get_response(user_prompt=prompt)
    print("\n--- LLM Output ---")
    print(response)

    # Example usage of the embed_text function
    text_to_embed = "This is a sample text for embedding."
    embedding = embed_text(text=text_to_embed)
    print("\n--- Embedding Output (first 5 values) ---")
    print(embedding[:5])