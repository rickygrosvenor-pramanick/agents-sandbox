"""
main.py

This script creates a FastAPI web server to expose the RAG agent's
functionality via an API endpoint. This is the first step in automating
the agent and making it available as a service.

curl -X 'POST' \
  'http://127.0.0.1:8000/generate-story' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "How did ING Group’s net interest income and cost/income ratio change between 3Q2022 and 3Q2023 across Retail Banking, Wholesale Banking, and the Corporate Line, and what might explain these differences?"
}'
"""
from fastapi import FastAPI
from pydantic import BaseModel
from agent import generate_enhanced_prompt
from retriever import query_vector_store
from utils.llm import get_response, llm, embedding_model

# Initialize the FastAPI app
app = FastAPI(
    title="Business Analyst User Story Agent",
    description="An agent that uses RAG to generate user stories from a knowledge base.",
    version="0.1.0",
)

class StoryRequest(BaseModel):
    """Defines the structure of the request body for the /generate-story endpoint."""
    prompt: str

class StoryResponse(BaseModel):
    """Defines the structure of the response for the /generate-story endpoint."""
    story: str

@app.post("/generate-story", response_model=StoryResponse)
def generate_story_endpoint(request: StoryRequest):
    """
    API endpoint to generate a user story.
    It takes a user prompt, runs the full RAG pipeline, and returns the
    generated story.
    """
    print(f"--- Received API Request for Prompt: '{request.prompt}' ---")

    # This is the same logic as in agent.py, but adapted for an API
    
    # 1. Retrieve context
    retrieved_results = query_vector_store(query=request.prompt, n_results=3)
    retrieved_docs = []
    if retrieved_results and retrieved_results.get('documents'):
        retrieved_docs = retrieved_results['documents'][0]

    # 2. Generate enhanced prompt
    enhanced_prompt = generate_enhanced_prompt(request.prompt, retrieved_docs)

    # 3. Get final response from the (mocked) LLM
    final_response = get_response(user_prompt=enhanced_prompt, llm=llm)

    return StoryResponse(story=final_response)

if __name__ == "__main__":
    import uvicorn
    print("--- Starting FastAPI Server ---")
    print("Access the API documentation at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
