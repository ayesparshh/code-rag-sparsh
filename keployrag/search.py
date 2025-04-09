import numpy as np
from keployrag.index import load_index, get_metadata
from keployrag.embeddings import generate_embeddings

def search_code(query, k=5):
    """Search the FAISS index using a text query."""
    try:
        index = load_index()  # Load the FAISS index
        
        if index.ntotal == 0:
            print("FAISS index is empty")
            return []
            
        query_embedding = generate_embeddings(query)

        # Perform the search in FAISS
        distances, indices = index.search(query_embedding, k)
        
        if len(indices) == 0 or len(indices[0]) == 0:
            print("No search results found")
            return []

        results = []
        metadata_list = get_metadata()
        
        if not metadata_list:
            print("Metadata is empty")
            return []

        for i, idx in enumerate(indices[0]):
            if idx < len(metadata_list):
                file_data = metadata_list[idx]
                results.append({
                    "filename": file_data["filename"],
                    "filepath": file_data["filepath"],
                    "content": file_data["content"],
                    "distance": distances[0][i]
                })

        return results
        
    except Exception as e:
        print(f"Error in search_code: {str(e)}")
        return []