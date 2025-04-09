import faiss
import os
from keployrag.index import load_index, retrieve_vectors, inspect_metadata, add_to_index, save_index, clear_index
from keployrag.embeddings import generate_embeddings
from keployrag.monitor import should_ignore_path

def test_faiss_index():
    # Clear the index before testing
    clear_index()
    
    # Walk through the codebase directory
    codebase_dir = os.getenv("WATCHED_DIR")
    files_processed = 0
    
    for root, _, files in os.walk(codebase_dir):
        if should_ignore_path(root):
            continue
            
        for file in files:
            if not file.endswith('.py'):
                continue
                
            filepath = os.path.join(root, file)
            if should_ignore_path(filepath):
                continue
                
            # Read and process each file
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Generate embeddings
                embeddings = generate_embeddings(file_content)
                if embeddings is None:
                    print(f"Embedding generation failed for {filepath}")
                    continue

                # Add to index
                add_to_index(embeddings, file_content, file, filepath)
                files_processed += 1
                print(f"Indexed file: {filepath}")
            
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
    
    save_index()
    print(f"Processed {files_processed} files")

    # Verify the index
    index = load_index()
    assert index.ntotal > 0, "FAISS index is empty!"
    print(f"FAISS index has {index.ntotal} vectors.")
    inspect_metadata(5)

if __name__ == "__main__":
    test_faiss_index()