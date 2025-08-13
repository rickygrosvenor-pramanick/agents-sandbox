"""
llm.py

A utility module for interacting with a LangChain LLM.
This provides a simple function to get a response from a pre-initialized model.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from typing import List

def get_response(user_prompt: str, llm: ChatOpenAI, oauth_token: str = None) -> str:
    """
    Gets a response from the provided LangChain ChatOpenAI model.
    NOTE: This is a MOCKED implementation for development purposes.

    Args:
        user_prompt: The user's input prompt as a string.
        llm: An initialized instance of ChatOpenAI.
        oauth_token: Placeholder for a token.

    Returns:
        A mock string response.
    """
    print("--- MOCKED LLM RESPONSE ---")
    print(f"Received prompt: '{user_prompt[:100]}...'")
    
    # The actual LLM call is commented out for mock mode.
    # if not isinstance(llm, ChatOpenAI):
    #     raise TypeError("The 'llm' parameter must be an instance of ChatOpenAI.")
    #
    # messages = [HumanMessage(content=user_prompt)]
    #
    # try:
    #     result: LLMResult = llm.generate([messages])
    #     response_content = result.generations[0][0].text
    #     return response_content
    # except Exception as e:
    #     print(f"An error occurred while communicating with the LLM: {e}")
    #     return "Sorry, I was unable to get a response from the model."

    mock_response = """
SUMMARY:
This is a mock summary for a user story about a new login feature.

DESCRIPTION:
As a user, I want to be able to log in with my email and password so that I can access my account securely. This is a mock description.

ACCEPTANCE CRITERIA:
```gherkin
Given I am on the login page
When I enter my valid credentials
And I click the "Login" button
Then I should be redirected to my dashboard
```

POINTS ESTIMATE: 5
"""
    return mock_response

def embed_text(text: str, embedding_model: OpenAIEmbeddings) -> List[float]:
    """
    Generates an embedding for the given text using the provided model.
    NOTE: This is a MOCKED implementation for development purposes.

    Args:
        text: The text to embed.
        embedding_model: An initialized instance of OpenAIEmbeddings.

    Returns:
        A mock embedding vector (list of floats).
    """
    print("--- MOCKED EMBEDDING ---")
    print(f"Received text to embed: '{text[:100]}...'")

    # The actual embedding call is commented out for mock mode.
    # if not isinstance(embedding_model, OpenAIEmbeddings):
    #     raise TypeError("The 'embedding_model' parameter must be an instance of OpenAIEmbeddings.")
    #
    # try:
    #     return embedding_model.embed_query(text)
    # except Exception as e:
    #     print(f"An error occurred while creating the embedding: {e}")
    #     return []

    # Return a fake vector of the correct dimension for text-embedding-3-small (1536)
    mock_vector = [0.01] * 1536
    mock_vector[0] = 0.99 # Make it identifiable
    return mock_vector

if __name__ == '__main__':
    # This is an example of how to use the mocked get_response function.
    # It does not require an API key.

    # 1. Initialize the LLM (can be None for the mock)
    # In a real scenario, this would be a configured ChatOpenAI object.
    my_llm = None 

    # 2. Define a prompt
    prompt = "Create a user story for a login feature."

    # 3. Get the response
    print(f"User Prompt: {prompt}")
    response = get_response(user_prompt=prompt, llm=my_llm)
    print("\n--- Mock LLM Output ---")
    print(response)

    # Example usage of the mocked embed_text function
    mock_embedding_model = None  # This would be an instance of OpenAIEmbeddings in a real scenario
    text_to_embed = "As a user, I want to reset my password."
    embedding = embed_text(text=text_to_embed, embedding_model=mock_embedding_model)
    print("\n--- Mock Embedding Output ---")
    print(embedding)