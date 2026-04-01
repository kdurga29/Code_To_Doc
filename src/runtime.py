"""Shared runtime builders for the documentation workflow."""

from __future__ import annotations

import os

try:
    from documentation import (
        HuggingFaceDocumentationGenerator,
        RepositoryDocumentationRAGChain,
        RepositoryDocumentationWorkflow,
    )
    from embeddings import (
        EmbeddingRuntimeConfig,
        LangChainEmbeddingModel,
        QdrantVectorStore,
        RepositoryCSTEmbeddingIndexer,
    )
    from parser import GitHubRepositoryExtractor, TreeSitterCodeParser
except ImportError:
    from src.documentation import (
        HuggingFaceDocumentationGenerator,
        RepositoryDocumentationRAGChain,
        RepositoryDocumentationWorkflow,
    )
    from src.embeddings import (
        EmbeddingRuntimeConfig,
        LangChainEmbeddingModel,
        QdrantVectorStore,
        RepositoryCSTEmbeddingIndexer,
    )
    from src.parser import GitHubRepositoryExtractor, TreeSitterCodeParser


def build_documentation_workflow(
    runtime_config: EmbeddingRuntimeConfig | None = None,
    github_token: str | None = None,
) -> RepositoryDocumentationWorkflow:
    """Construct the reusable index-and-document workflow from runtime settings."""

    config = runtime_config or EmbeddingRuntimeConfig.from_env()
    extractor = GitHubRepositoryExtractor(token=github_token or os.getenv("GITHUB_TOKEN") or None)
    parser = TreeSitterCodeParser()
    embedder = LangChainEmbeddingModel(
        model_name=config.model_name,
        token=config.hf_token,
    )
    vector_store = QdrantVectorStore(
        collection_name=config.qdrant_collection_name,
        url=config.qdrant_url,
        qdrant_api_key=config.qdrant_api_key,
    )
    indexer = RepositoryCSTEmbeddingIndexer(
        extractor=extractor,
        code_parser=parser,
        embedder=embedder,
        vector_store=vector_store,
    )
    rag_chain = RepositoryDocumentationRAGChain(
        embedder=embedder,
        vector_store=vector_store,
        generator=HuggingFaceDocumentationGenerator(
            model_name=config.docs_model_name,
            hf_token=config.hf_token,
            max_new_tokens=config.docs_max_new_tokens,
            temperature=config.docs_temperature,
            repetition_penalty=config.docs_repetition_penalty,
            load_in_4bit=config.docs_load_in_4bit,
        ),
        retrieval_limit=config.docs_retrieval_limit,
        context_char_limit=config.docs_context_char_limit,
    )
    return RepositoryDocumentationWorkflow(
        indexer=indexer,
        rag_chain=rag_chain,
        vector_store=vector_store,
    )
