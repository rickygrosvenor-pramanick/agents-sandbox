"""
retriever.py

This script handles the retrieval of relevant documents from the ChromaDB
vector store based on a user's query. It performs the following steps:
1.  Initializes a connection to the existing ChromaDB.
2.  Takes a user query as input.
3.  Generates an embedding for the query using the same (mock) model used
    during ingestion.
4.  Queries the vector store to find the most similar document chunks.
5.  Returns the retrieved chunks, which can then be used as context for an LLM.
"""
import os
import chromadb
from typing import List, Dict

# Import our custom utilities
from utils.llm import embed_text # Using our mock embedding function

# --- Constants ---
DB_DIR = os.path.join(os.path.dirname(__file__), 'db')
CHROMA_COLLECTION_NAME = "rag_collection"

def query_vector_store(query: str, n_results: int = 5) -> Dict:
    """
    Queries the ChromaDB collection to find documents relevant to the user's query.

    Args:
        query: The user's question or query string.
        n_results: The number of results to retrieve.

    Returns:
        A dictionary containing the query results from ChromaDB.
    """
    if not os.path.isdir(DB_DIR):
        print(f"Error: Database directory not found at '{DB_DIR}'.")
        print("Please run the 'ingest.py' script first to create the database.")
        return {}

    print("--- Querying Vector Store ---")

    # 1. Initialize ChromaDB client and get the collection
    print(f"Connecting to ChromaDB at '{DB_DIR}'...")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"Successfully connected to collection '{CHROMA_COLLECTION_NAME}'.")
    except ValueError:
        print(f"Error: Collection '{CHROMA_COLLECTION_NAME}' not found.")
        print("Please ensure you have ingested data using 'ingest.py'.")
        return {}


    # 2. Generate an embedding for the user's query
    print(f"Generating embedding for query: '{query}'")
    query_embedding = embed_text(text=query) # Mock embedding call

    # 3. Query the collection
    print(f"Performing query to find top {n_results} results...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    print("--- Query Complete ---")
    return results

if __name__ == '__main__':
    # Example of how to use the retriever
    sample_query = "What were the net profits for the last quarter?"

    print(f"Executing sample query: '{sample_query}'")
    
    retrieved_results = query_vector_store(query=sample_query, n_results=3)

    if retrieved_results:
        print("\n--- Retrieved Results ---")
        
        # Because we are using a MOCK embedding function that returns the same
        # vector for every piece of text, the results will NOT be semantically
        # relevant to the query. They are effectively random. This demonstrates
        # that the retrieval mechanism is working, but meaningful results
        # require a real embedding model.
        
        ids = retrieved_results.get('ids', [[]])[0]
        documents = retrieved_results.get('documents', [[]])[0]
        metadatas = retrieved_results.get('metadatas', [[]])[0]
        distances = retrieved_results.get('distances', [[]])[0]

        if not documents:
            print("The query returned no documents.")
        else:
            for i, doc in enumerate(documents):
                print(f"\n--- Result {i+1} (Distance: {distances[i]:.4f}) ---")
                print(f"  Source: {metadatas[i].get('source', 'N/A')}")
                print(f"  Type:   {metadatas[i].get('content_type', 'N/A')}")
                print(f"  Content: {doc[:500]}...") # Print first 500 chars
    else:
        print("Could not retrieve results.")
