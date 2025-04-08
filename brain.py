import os
from io import BytesIO
from typing import Tuple, List, Dict
import time
from pathlib import Path

from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

# Global variables for tracking document state
document_timestamps: Dict[str, float] = {}
current_index = None
current_embeddings = None

def parse_mdx(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    content = file.read().decode('utf-8')
    # You might want to add more sophisticated MDX parsing here
    # print("Parsing MDX file:", filename)
    return [content], filename 

def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    doc_chunks = []
    for i, page in enumerate(text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        chunks = text_splitter.split_text(page)
        for j, chunk in enumerate(chunks):
            # Calculate the start and end line numbers for this chunk
            start_line = page[:page.index(chunk)].count('\n') + 1
            end_line = start_line + chunk.count('\n')
            
            doc = Document(
                page_content=chunk, 
                metadata={
                    "chunk": j,
                    "source": f"{filename}:{start_line}-{end_line}",
                    "filename": filename,
                    "start_line": start_line,
                    "end_line": end_line
                }
            )
            doc_chunks.append(doc)
    return doc_chunks

def get_file_timestamp(filepath: str) -> float:
    """Get the last modification timestamp of a file."""
    return os.path.getmtime(filepath)

def needs_update(filepath: str) -> bool:
    """Check if a file needs to be updated in the index."""
    if filepath not in document_timestamps:
        return True
    return get_file_timestamp(filepath) > document_timestamps[filepath]

def update_document_index(filepath: str, index: FAISS, embeddings: AzureOpenAIEmbeddings) -> FAISS:
    """Update a single document in the index."""
    try:
        with open(filepath, "rb") as f:
            content = f.read()
        
        text, filename = parse_mdx(BytesIO(content), os.path.basename(filepath))
        documents = text_to_docs(text, filename)
        
        # Remove old embeddings for this file if they exist
        # Note: FAISS doesn't support direct deletion, so we need to rebuild the index
        # for the specific document
        
        new_index = FAISS.from_documents(documents, embeddings)
        
        # Merge the new embeddings with the existing index
        index.merge_from(new_index)
        
        # Update timestamp
        document_timestamps[filepath] = get_file_timestamp(filepath)
        
        return index
    except Exception as e:
        print(f"Error updating document {filepath}: {str(e)}")
        return index

def get_index_for_mdx(mdx_files, mdx_names):
    global current_index, current_embeddings
    print("Creating/updating index for MDX files...")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        model="keploy-docs-embedding", # text-embedding-ada-002
        chunk_size=1,
    )
    
    if os.path.exists("document_index") and current_index:
        print("Using existing index and checking for updates...")
        index = current_index
    else:
        if os.path.exists("document_index"):
            print("Loading existing index...")
            index = FAISS.load_local(
                folder_path="document_index",
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new index...")
            documents = []
            for mdx_file, mdx_name in zip(mdx_files, mdx_names):
                text, filename = parse_mdx(BytesIO(mdx_file), mdx_name)
                documents.extend(text_to_docs(text, filename))
            index = FAISS.from_documents(documents, embeddings)

    # Store current state
    current_index = index
    current_embeddings = embeddings

    # Save the index
    index.save_local("document_index")
    return index