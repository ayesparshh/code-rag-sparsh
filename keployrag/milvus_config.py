from pymilvus import connections
import time
import sys

# Milvus connection settings
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "code_embeddings"

# Vector parameters
EMBEDDING_DIM = 1536
INDEX_TYPE = "HNSW"
METRIC_TYPE = "L2"
INDEX_PARAMS = {
    "M": 16,  # Number of bi-directional links
    "efConstruction": 200  # Index time accuracy vs speed trade-off
}
SEARCH_PARAMS = {
    "ef": 50  # Query time accuracy vs speed trade-off
}

def connect_milvus(max_retries=3, retry_delay=5):
    """Connect to Milvus server with retries"""
    for attempt in range(max_retries):
        try:
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
            print("Successfully connected to Milvus")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to connect to Milvus after maximum retries")
                print("Please ensure Milvus server is running (docker-compose up -d)")
                return False