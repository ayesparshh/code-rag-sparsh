import os
import logging
import warnings
from keployrag.index import clear_index, add_to_index
from keployrag.embeddings import generate_embeddings
from keployrag.config import WATCHED_DIR
from keployrag.monitor import start_monitoring, should_ignore_path
from keployrag.milvus_config import connect_milvus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def full_reindex():
    """Perform a full reindex of the codebase using Milvus"""
    logging.info("Starting full reindexing of the codebase...")
    files_processed = 0
    
    # Ensure Milvus connection
    connect_milvus()
    
    for root, _, files in os.walk(WATCHED_DIR):
        if should_ignore_path(root):
            logging.info(f"Ignoring directory: {root}")
            continue

        for file in files:
            filepath = os.path.join(root, file)
            if should_ignore_path(filepath) or not file.endswith(".py"):
                continue

            logging.info(f"Processing file: {filepath}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    full_content = f.read()

                embeddings = generate_embeddings(full_content)
                if embeddings is not None:
                    add_to_index(embeddings, full_content, file, filepath)
                    files_processed += 1
                else:
                    logging.warning(f"Failed to generate embeddings for {filepath}")
            except Exception as e:
                logging.error(f"Error processing file {filepath}: {e}")

    logging.info(f"Full reindexing completed. {files_processed} files processed.")

def main():
    # Initialize Milvus connection
    connect_milvus()
    
    # Clear and recreate the collection
    clear_index()

    # Perform full reindex
    full_reindex()

    # Start monitoring
    start_monitoring()

if __name__ == "__main__":
    main()