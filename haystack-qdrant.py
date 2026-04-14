"""RAG pipeline using Haystack, Qdrant, and Azure Files.

Ingests documents from an Azure file share, converts files to Haystack
Documents, indexes into Qdrant via a Haystack indexing pipeline, and
provides an interactive Q&A loop.
"""

import os
import tempfile
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", message="Mutating attribute")
warnings.filterwarnings("ignore", message="has metadata fields with unsupported types")
warnings.filterwarnings("ignore", message="required_variables.*is not set")

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import (
    CSVToDocument,
    DOCXToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.embedders import (
    AzureOpenAIDocumentEmbedder,
    AzureOpenAITextEmbedder,
)
from haystack.components.generators import AzureOpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from azure_files import DownloadedFile, connect_to_share, download_files, list_share_files
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CREDENTIAL,
    EMBEDDING_DIMENSIONS,
    OPENAI_CHAT_DEPLOYMENT,
    OPENAI_EMBEDDING_DEPLOYMENT,
    OPENAI_ENDPOINT,
    SHARE_NAME,
    STORAGE_ACCOUNT_NAME,
    TOKEN_PROVIDER,
)

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "azure-files-rag")

# Mapping of file extensions to Haystack document converters
CONVERTER_MAP: dict = {
    ".pdf": PyPDFToDocument(),
    ".docx": DOCXToDocument(),
    ".csv": CSVToDocument(),
}

DEFAULT_CONVERTER = TextFileToDocument()


def parse_downloaded_files(
    downloaded_files: list[DownloadedFile],
) -> list[Document]:
    """Parse downloaded files from an Azure file share into Haystack Documents.

    Args:
        downloaded_files: A list of DownloadedFile objects, each containing
            the path and access control metadata for a file.

    Returns: A list of Haystack Documents.
    """
    documents = []

    for info in downloaded_files:
        file_ext = os.path.splitext(info.file_name.lower())[1]
        converter = CONVERTER_MAP.get(file_ext, DEFAULT_CONVERTER)

        metadata = {
            "azure_file_path": info.relative_path,
            "file_name": info.file_name,
        }

        try:
            result = converter.run(sources=[Path(info.local_path)], meta=metadata)
            docs = result["documents"]
        except Exception:
            print(f"Failed to parse {info.relative_path}, skipping...")
            continue

        documents.extend(docs)

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding.

    Args:
        documents: A list of Haystack Documents to split.

    Returns: A list of smaller Document chunks with preserved metadata.
    """
    splitter = DocumentSplitter(
        split_by="word",
        split_length=CHUNK_SIZE,
        split_overlap=CHUNK_OVERLAP,
    )
    result = splitter.run(documents=documents)
    return result["documents"]


def embed_and_index(chunks: list[Document]) -> QdrantDocumentStore:
    """Embed document chunks via Azure OpenAI and upsert into a Qdrant collection.

    Args:
        chunks: A list of chunked Haystack Documents to embed and index.

    Returns: A QdrantDocumentStore connected to the populated collection.
    """
    # Reset collection if specified via environment variable. This is useful
    # for development to ensure a clean slate on each run. In production, you
    # would typically not reset the collection.
    recreate = os.getenv("RESET_INDEX") == "true"

    store = QdrantDocumentStore(
        url=QDRANT_URL,
        api_key=Secret.from_env_var("QDRANT_API_KEY"),
        index=QDRANT_COLLECTION_NAME,
        embedding_dim=EMBEDDING_DIMENSIONS,
        similarity="cosine",
        wait_result_from_api=True,
        recreate_index=recreate,
    )

    embedder = AzureOpenAIDocumentEmbedder(
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=OPENAI_EMBEDDING_DEPLOYMENT,
        azure_ad_token_provider=TOKEN_PROVIDER,
        api_key=None,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    writer = DocumentWriter(
        document_store=store,
        policy=DuplicatePolicy.OVERWRITE,
    )

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", embedder)
    indexing_pipeline.add_component("writer", writer)
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    indexing_pipeline.run({"embedder": {"documents": chunks}})

    return store


_PROMPT_TEMPLATE = """\
Answer the question based on the context below. \
Be specific and cite the source file name in brackets for each fact.

{% for doc in documents %}
[{{ doc.meta.get("azure_file_path", "") }}]
{{ doc.content }}

{% endfor %}

Question: {{ query }}

Answer:"""


def build_query_pipeline(
    document_store: QdrantDocumentStore,
):
    """Build a retrieval Q&A pipeline.

    Constructs a Haystack Pipeline that embeds the user query, retrieves
    relevant documents from Qdrant, and generates an answer via Azure OpenAI.

    Args:
        document_store: QdrantDocumentStore to retrieve from.

    Returns: A callable that takes a question string and returns an answer.
    """
    text_embedder = AzureOpenAITextEmbedder(
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=OPENAI_EMBEDDING_DEPLOYMENT,
        azure_ad_token_provider=TOKEN_PROVIDER,
        api_key=None,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    retriever = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=5,
    )

    prompt_builder = PromptBuilder(template=_PROMPT_TEMPLATE)

    generator = AzureOpenAIGenerator(
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=OPENAI_CHAT_DEPLOYMENT,
        azure_ad_token_provider=TOKEN_PROVIDER,
        api_key=None,
    )

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.add_component("prompt_builder", prompt_builder)
    query_pipeline.add_component("generator", generator)

    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever.documents", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder.prompt", "generator.prompt")

    def run_query(question: str) -> str:
        """Run a question through the query pipeline."""
        result = query_pipeline.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"query": question},
        })
        replies = result["generator"]["replies"]
        return replies[0] if replies else "No answer generated."

    return run_query


def main():
    """Main execution flow."""
    share = connect_to_share(STORAGE_ACCOUNT_NAME, SHARE_NAME, CREDENTIAL)

    # 1. List files from the share
    print("Scanning file share...")
    file_references = list_share_files(share)
    if not file_references:
        print("No files found.")
        return
    print(f"Found {len(file_references)} files.\n")

    # 2. Download files (shared Azure Files logic)
    print("Downloading files onto temporary local directory...")
    with tempfile.TemporaryDirectory() as temp_directory:
        downloaded = download_files(file_references, temp_directory)
        if not downloaded:
            print("No files downloaded.")
            return
        print()

        # 3. Parse into Haystack Documents
        print("Parsing files...")
        documents = parse_downloaded_files(downloaded)

    if not documents:
        print("No documents parsed.")
        return
    print(f"{len(documents)} documents.\n")

    # 4. Chunk
    print("Splitting into chunks...")
    chunks = chunk_documents(documents)
    print(f"{len(documents)} docs -> {len(chunks)} chunks.\n")

    # 5. Embed and index
    print("Indexing into Qdrant...")
    store = embed_and_index(chunks)
    print(f"{len(chunks)} chunks indexed.\n")

    run_query = build_query_pipeline(store)
    print("Ready. Type 'quit' to exit.\n")

    try:
        while True:
            question = input("You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            print(f"\nAnswer: {run_query(question)}\n")
    except KeyboardInterrupt:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
