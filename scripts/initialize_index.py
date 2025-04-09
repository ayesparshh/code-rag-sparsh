from keployrag.index import clear_index
from keployrag.milvus_config import connect_milvus

def initialize_index():
    """Initialize Milvus collection"""
    connect_milvus()
    clear_index()
    print("Milvus collection initialized.")

if __name__ == "__main__":
    initialize_index()