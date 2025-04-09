import numpy as np
from keployrag.index import search_code as milvus_search_code
from keployrag.embeddings import generate_embeddings

def search_code(query, k=5):
    """Search the Milvus collection using a text query."""
    try:
        # Generate embeddings for the query
        query_embedding = generate_embeddings(query)
        
        if query_embedding is None:
            print("Failed to generate query embedding")
            return []
            
        # Use the Milvus search function directly
        results = milvus_search_code(query_embedding, k=k)
        return results
        print(f"hi")
    except Exception as e:
        print(f"Error in search_code: {str(e)}")
        print("Please ensure Milvus server is running (docker-compose up -d)")
        print("does this index?")
        return []