import numpy as np
from keployrag.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_EMBEDDING_DEPLOYMENT
)
from openai import AzureOpenAI

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def generate_embeddings(text):
    """Generate embeddings using Azure OpenAI."""
    try:
        response = client.embeddings.create(
            model=AZURE_EMBEDDING_DEPLOYMENT,
            input=[text]
        )
        embeddings = response.data[0].embedding
        return np.array(embeddings).astype('float32').reshape(1, -1)
    except Exception as e:
        print(f"Error generating embeddings with Azure OpenAI: {e}")
        return None