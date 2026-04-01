from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from documentation import (
    HuggingFaceDocumentationGenerator,
    RepositoryDocumentationRAGChain,
    RepositoryDocumentationResult,
    RepositoryDocumentationWorkflow,
)
from embeddings import QdrantVectorStore, RepositoryEmbeddingResult, RetrievedCSTDocument


def test_repository_exists_checks_qdrant_with_repository_filter() -> None:
    fake_client = Mock()
    fake_client.collection_exists.return_value = True
    fake_client.scroll.return_value = ([SimpleNamespace(id="doc-1")], None)

    filter_ctor = Mock(side_effect=lambda must: {"must": must})
    field_condition_ctor = Mock(side_effect=lambda **kwargs: kwargs)
    match_value_ctor = Mock(side_effect=lambda value: {"value": value})
    fake_models = SimpleNamespace(
        Filter=filter_ctor,
        FieldCondition=field_condition_ctor,
        MatchValue=match_value_ctor,
    )

    store = QdrantVectorStore(
        collection_name="repo-cst",
        client=fake_client,
        models_module=fake_models,
    )

    assert store.repository_exists("acme/repo", ref="main") is True
    fake_client.scroll.assert_called_once()
    assert field_condition_ctor.call_args_list[0].kwargs["key"] == "repository"
    assert field_condition_ctor.call_args_list[1].kwargs["key"] == "ref"


def test_search_repository_documents_returns_ranked_payloads() -> None:
    fake_client = Mock()
    fake_client.collection_exists.return_value = True
    fake_client.search.return_value = [
        SimpleNamespace(
            id="doc-1",
            score=0.87,
            payload={
                "path": "src/main.py",
                "language": "python",
                "repository": "acme/repo",
                "ref": "main",
                "cst": "(module)",
                "root_type": "module",
            },
        )
    ]

    fake_models = SimpleNamespace(
        Filter=lambda must: {"must": must},
        FieldCondition=lambda **kwargs: kwargs,
        MatchValue=lambda value: {"value": value},
    )

    store = QdrantVectorStore(
        collection_name="repo-cst",
        client=fake_client,
        models_module=fake_models,
    )

    documents = store.search_repository_documents(
        repository="acme/repo",
        ref="main",
        query_vector=[0.1, 0.2],
        limit=3,
    )

    assert len(documents) == 1
    assert documents[0].doc_id == "doc-1"
    assert documents[0].path == "src/main.py"
    assert documents[0].metadata["root_type"] == "module"
    fake_client.search.assert_called_once()


def test_hf_documentation_generator_invokes_local_model_with_prompt() -> None:
    llm = Mock()
    llm.invoke.return_value = "# Generated Docs"

    generator = HuggingFaceDocumentationGenerator(
        model_name="deepseek-ai/deepseek-coder-1.3b-instruct",
        llm=llm,
    )

    markdown = generator.generate_markdown(
        repository="acme/repo",
        ref="main",
        retrieved_documents=[
            RetrievedCSTDocument(
                doc_id="doc-1",
                path="src/main.py",
                language="python",
                cst="(module)",
                metadata={"root_type": "module"},
                score=0.9,
            )
        ],
    )

    assert markdown == "# Generated Docs"
    llm.invoke.assert_called_once()
    prompt = llm.invoke.call_args.args[0]
    assert "Repository: acme/repo" in prompt
    assert "Reference: main" in prompt
    assert "src/main.py" in prompt


def test_hf_documentation_generator_supports_structured_responses() -> None:
    llm = Mock()
    llm.invoke.return_value = [{"generated_text": "# API Docs"}]

    generator = HuggingFaceDocumentationGenerator(llm=llm)

    markdown = generator.generate_markdown(
        repository="acme/repo",
        ref=None,
        retrieved_documents=[
            RetrievedCSTDocument(
                doc_id="doc-1",
                path="src/main.py",
                language="python",
                cst="(module)",
                metadata={"root_type": "module"},
                score=0.9,
            )
        ],
    )

    assert markdown == "# API Docs"


def test_rag_chain_retrieves_deduplicated_context_and_generates_markdown() -> None:
    embedder = Mock()
    embedder.embed_documents.return_value = [[0.1, 0.2]]

    doc_a = RetrievedCSTDocument(
        doc_id="doc-a",
        path="src/main.py",
        language="python",
        cst="(module)",
        metadata={"root_type": "module"},
        score=0.94,
    )
    doc_b = RetrievedCSTDocument(
        doc_id="doc-b",
        path="src/utils.py",
        language="python",
        cst="(module (function_definition))",
        metadata={"root_type": "module"},
        score=0.88,
    )

    vector_store = Mock()
    vector_store.repository_exists.return_value = True
    vector_store.search_repository_documents.side_effect = [
        [doc_a, doc_b],
        [doc_a],
        [],
        [],
    ]

    generator = Mock()
    generator.generate_markdown.return_value = "# acme/repo"

    chain = RepositoryDocumentationRAGChain(
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        retrieval_limit=4,
    )

    result = chain.generate("https://github.com/acme/repo", ref="main")

    assert result.repository == "acme/repo"
    assert result.markdown == "# acme/repo"
    assert [document.doc_id for document in result.retrieved_documents] == ["doc-a", "doc-b"]
    assert embedder.embed_documents.call_count == 4
    generator.generate_markdown.assert_called_once()


def test_workflow_indexes_repository_before_generating_when_missing() -> None:
    vector_store = Mock()
    vector_store.repository_exists.return_value = False

    index_result = RepositoryEmbeddingResult(
        repository="acme/repo",
        ref="main",
        fetched_files=2,
        parsed_files=[],
        skipped_files=[],
        stored_ids=["doc-1"],
        collection_name="repo-cst",
    )
    indexer = Mock()
    indexer.index_repository.return_value = index_result

    documentation = RepositoryDocumentationResult(
        repository="acme/repo",
        ref="main",
        markdown="# Docs",
        retrieved_documents=[],
    )
    rag_chain = Mock()
    rag_chain.generate.return_value = documentation

    workflow = RepositoryDocumentationWorkflow(
        indexer=indexer,
        rag_chain=rag_chain,
        vector_store=vector_store,
    )

    result = workflow.run("https://github.com/acme/repo", ref="main")

    assert result.already_indexed is False
    assert result.indexed_now is True
    assert result.index_result == index_result
    indexer.index_repository.assert_called_once_with("https://github.com/acme/repo", ref="main")
    rag_chain.generate.assert_called_once_with("https://github.com/acme/repo", ref="main")


def test_workflow_skips_indexing_when_repository_already_exists() -> None:
    vector_store = Mock()
    vector_store.repository_exists.return_value = True

    indexer = Mock()
    documentation = RepositoryDocumentationResult(
        repository="acme/repo",
        ref=None,
        markdown="# Docs",
        retrieved_documents=[],
    )
    rag_chain = Mock()
    rag_chain.generate.return_value = documentation

    workflow = RepositoryDocumentationWorkflow(
        indexer=indexer,
        rag_chain=rag_chain,
        vector_store=vector_store,
    )

    result = workflow.run("acme/repo")

    assert result.already_indexed is True
    assert result.indexed_now is False
    assert result.index_result is None
    indexer.index_repository.assert_not_called()
    rag_chain.generate.assert_called_once_with("acme/repo", ref=None)
