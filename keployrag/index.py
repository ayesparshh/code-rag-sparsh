from pymilvus import (
    Collection,
    CollectionSchema, 
    FieldSchema,
    DataType,
    connections,
    utility
)
import numpy as np
import os
from keployrag.config import EMBEDDING_DIM, WATCHED_DIR
from keployrag.milvus_config import (
    COLLECTION_NAME,
    INDEX_TYPE,
    METRIC_TYPE,
    INDEX_PARAMS,
    SEARCH_PARAMS,
    connect_milvus
)

def create_collection():
    """Create Milvus collection with schema"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="filepath", dtype=DataType.VARCHAR, max_length=1024)
    ]
    schema = CollectionSchema(fields=fields, description="Code embeddings collection")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    # Create HNSW index
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": INDEX_TYPE,
            "metric_type": METRIC_TYPE,
            "params": INDEX_PARAMS
        }
    )
    return collection

def clear_index():
    """Drop the collection if it exists and recreate it"""
    connect_milvus()
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    create_collection()
    print("Milvus collection cleared and reinitialized.")

def add_to_index(embeddings, full_content, filename, filepath):
    """Add document to Milvus collection"""
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    # Convert to relative filepath
    relative_filepath = os.path.relpath(filepath, WATCHED_DIR)
    
    # Prepare data
    entities = [
        {
            "embedding": embeddings[0].tolist(),
            "content": full_content,
            "filename": filename,
            "filepath": relative_filepath
        }
    ]
    
    # Insert data
    collection.insert(entities)
    collection.flush()

def save_index():
    """Save collection state - in Milvus this is handled automatically"""
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.flush()
    return True    

def load_index():
    """Load collection state and return the collection object"""
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.load()
    return collection

def get_metadata():
    """Get all metadata from collection"""
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    results = collection.query(
        expr="id >= 0",
        output_fields=["content", "filename", "filepath"]
    )
    
    return results

def retrieve_vectors(n=5):
    """Retrieve the first n vectors from collection"""
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    results = collection.query(
        expr=f"id < {n}",
        output_fields=["embedding"]
    )
    
    vectors = np.zeros((min(n, len(results)), EMBEDDING_DIM), dtype=np.float32)
    for i, result in enumerate(results):
        vectors[i] = np.array(result["embedding"])
    
    return vectors

def inspect_metadata(n=5):
    """Inspect the first n metadata entries"""
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    results = collection.query(
        expr=f"id < {n}",
        output_fields=["content", "filename", "filepath"]
    )
    
    print(f"Inspecting the first {n} metadata entries:")
    for i, data in enumerate(results):
        print(f"Entry {i}:")
        print(f"Filename: {data['filename']}")
        print(f"Filepath: {data['filepath']}")
        print(f"Content: {data['content'][:100]}...")  # Show first 100 characters
        print()

def search_code(query_embedding, k=5, filter_expr=None):
    """Search similar code in Milvus with optional metadata filtering"""
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    search_params = {"params": SEARCH_PARAMS}
    results = collection.search(
        data=[query_embedding[0].tolist()],
        anns_field="embedding",
        param=search_params,
        limit=k,
        expr=filter_expr,  # Add metadata filtering
        output_fields=["content", "filename", "filepath"]
    )
    
    return [
        {
            "content": hit.entity.get("content"),
            "filename": hit.entity.get("filename"),
            "filepath": hit.entity.get("filepath"),
            "distance": hit.distance,
            "id": hit.id  # Include ID in results
        }
        for hit in results[0]
    ]

def query_by_ids(ids, output_fields=None):
    """Query entities by their IDs"""
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    if output_fields is None:
        output_fields = ["content", "filename", "filepath"]
    
    if isinstance(ids, (int, str)):
        ids = [ids]
    
    # Create ID expression
    id_expr = f"id in {ids}"
    
    results = collection.query(
        expr=id_expr,
        output_fields=output_fields
    )
    
    return results

def delete_entities(ids=None, filter_expr=None):
    """Delete entities by IDs or filter expression"""
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    
    if ids is not None:
        if isinstance(ids, (int, str)):
            ids = [ids]
        expr = f"id in {ids}"
    elif filter_expr is not None:
        expr = filter_expr
    else:
        raise ValueError("Either ids or filter_expr must be provided")
    
    # Delete entities
    collection.delete(expr)
    collection.flush()
    
    return True

def get_collection_stats():
    """Get statistics about the collection"""
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    
    stats = {
        "name": collection.name,
        "description": collection.description,
        "num_entities": collection.num_entities,
        "schema": collection.schema,
        "indexes": collection.indexes
    }
    
    return stats