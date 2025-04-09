import os
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4")

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))

# Project directory
WATCHED_DIR = os.getenv("WATCHED_DIR", os.path.join(os.getcwd(), 'keployrag'))

# FAISS index file path
FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", os.path.join(WATCHED_DIR, 'keployrag_index.faiss'))

# Project-Specific Configuration
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

IGNORE_PATHS = [
    os.path.join(WATCHED_DIR, ".venv"),
    os.path.join(WATCHED_DIR, "node_modules"),
    os.path.join(WATCHED_DIR, "__pycache__"),
    os.path.join(WATCHED_DIR, ".git"),
    os.path.join(WATCHED_DIR, "tests"),
]