"""
ingest.py

This script handles the ingestion of documents from the corpus into a ChromaDB
vector store. It performs the following steps:
1.  Scans the `corpus` directory for supported documents.
2.  Uses the `data_scraper` utility to extract content (text and tables).
3.  Chunks the extracted content into manageable pieces.
4.  Generates embeddings for each chunk using a mock embedding function.
5.  Stores the chunks and their corresponding embeddings in a local ChromaDB
    collection for later retrieval.
"""
import os
import uuid
from tqdm import tqdm
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.documents.elements import Table, Text

# Import our custom utilities
from utils.data_scraper import scrape_document
from utils.llm import embed_text # Using our mock embedding function

# --- Constants ---
CORPUS_DIR = os.path.join(os.path.dirname(__file__), 'corpus')
DB_DIR = os.path.join(os.path.dirname(__file__), 'db')
CHROMA_COLLECTION_NAME = "rag_collection"

def ingest_data():
    """
    Main function to orchestrate the data ingestion pipeline.
    """
    if not os.path.isdir(CORPUS_DIR):
        print(f"Error: Corpus directory not found at '{CORPUS_DIR}'")
        return

    print("--- Starting Data Ingestion ---")

    # 1. Initialize ChromaDB client and collection
    print(f"Initializing ChromaDB client (persisting to '{DB_DIR}')...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    print(f"Collection '{CHROMA_COLLECTION_NAME}' ready.")

    # 2. Initialize Text Splitter
    # This helps break down long text into smaller, more manageable chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    # 3. Scan and process files from the corpus
    files_to_process = [f for f in os.listdir(CORPUS_DIR) if f.endswith(('.pdf', '.xlsx'))]
    print(f"Found {len(files_to_process)} documents to process in '{CORPUS_DIR}'.")

    for file_name in tqdm(files_to_process, desc="Ingesting Documents"):
        file_path = os.path.join(CORPUS_DIR, file_name)
        
        # Scrape the document to get a list of elements (text and tables)
        elements = scrape_document(file_path)

        if not elements:
            tqdm.write(f"Warning: No content extracted from {file_name}. Skipping.")
            continue

        for element in elements:
            # For tables, we use the HTML representation to preserve structure.
            if isinstance(element, Table) and hasattr(element, "metadata") and element.metadata.text_as_html:
                content = element.metadata.text_as_html
                content_type = "table"
            # For text, we use the plain text.
            elif isinstance(element, Text):
                content = element.text
                content_type = "text"
            else:
                continue # Skip elements we can't process

            if not content or not content.strip():
                continue

            # Chunk the content
            chunks = text_splitter.split_text(content)
            
            # Embed and store each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_name}_{element.id}_{i}"
                embedding = embed_text(text=chunk) # Mock embedding call
                
                metadata = {
                    "source": file_name,
                    "content_type": content_type,
                    "element_id": str(element.id)
                }
                
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[metadata]
                )

    print("\n--- Data Ingestion Complete ---")
    print(f"Total documents in collection: {collection.count()}")

if __name__ == '__main__':
    ingest_data()
