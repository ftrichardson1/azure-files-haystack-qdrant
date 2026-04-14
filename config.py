"""Shared configuration file"""

import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv()

# Azure Storage
STORAGE_ACCOUNT_NAME = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
SHARE_NAME = os.environ["AZURE_STORAGE_SHARE_NAME"]

# Azure OpenAI
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
OPENAI_CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

# RAG tuning
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# Azure authentication
CREDENTIAL = DefaultAzureCredential()
TOKEN_PROVIDER = get_bearer_token_provider(
    CREDENTIAL, "https://cognitiveservices.azure.com/.default"
)
