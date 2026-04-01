"""CST embedding generation and Qdrant indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import uuid
from typing import Any, TypedDict
import importlib
import os
from dotenv import load_dotenv
from qdrant_client import models

load_dotenv()

try:
    from parser import (
        GitHubRepositoryExtractor,
        ParsedFile,
        RepositoryFile,
        TreeSitterCodeParser,
    )
except ImportError:
    from src.parser import (
        GitHubRepositoryExtractor,
        ParsedFile,
        RepositoryFile,
        TreeSitterCodeParser,
    )


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_DOCUMENTATION_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    """Environment-driven settings for embedding generation and storage."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    hf_token: str | None = None
    qdrant_collection_name: str = "code_to_doc_cst"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    docs_model_name: str = DEFAULT_DOCUMENTATION_MODEL
    docs_max_new_tokens: int = 512
    docs_temperature: float = 0.1
    docs_repetition_penalty: float = 1.1
    docs_load_in_4bit: bool = True
    docs_retrieval_limit: int = 8
    docs_context_char_limit: int = 16_000

    @classmethod
    def from_env(cls) -> "EmbeddingRuntimeConfig":
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL),
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "code_to_doc_cst"),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            docs_model_name=os.getenv("DOCUMENTATION_MODEL_NAME", DEFAULT_DOCUMENTATION_MODEL),
            docs_max_new_tokens=int(os.getenv("DOCUMENTATION_MAX_NEW_TOKENS", "512")),
            docs_temperature=float(os.getenv("DOCUMENTATION_TEMPERATURE", "0.1")),
            docs_repetition_penalty=float(os.getenv("DOCUMENTATION_REPETITION_PENALTY", "1.1")),
            docs_load_in_4bit=_env_flag("DOCUMENTATION_LOAD_IN_4BIT", default=True),
            docs_retrieval_limit=int(os.getenv("DOCUMENTATION_RETRIEVAL_LIMIT", "8")),
            docs_context_char_limit=int(os.getenv("DOCUMENTATION_CONTEXT_CHAR_LIMIT", "16000")),
        )


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CSTDocument:
    """Embedding payload for one parsed file."""

    doc_id: str
    path: str
    language: str
    cst: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievedCSTDocument:
    """Retrieved vector-store payload for downstream RAG generation."""

    doc_id: str
    path: str
    language: str
    cst: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class RepositoryEmbeddingResult:
    """Result of parsing + embedding + vector indexing."""

    repository: str
    ref: str | None
    fetched_files: int
    parsed_files: list[ParsedFile]
    skipped_files: list[str]
    stored_ids: list[str]
    collection_name: str


class EmbeddingWorkflowState(TypedDict, total=False):
    """LangGraph state for repository embedding workflow."""

    repository_url: str
    ref: str | None
    repository_name: str
    files: list[RepositoryFile]
    parsed_files: list[ParsedFile]
    skipped_files: list[str]
    documents: list[CSTDocument]
    embeddings: list[list[float]]
    stored_ids: list[str]


class LangChainEmbeddingModel:
    """LangChain Hugging Face embedding wrapper."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        token: str | None = None,
        embeddings: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._token = token
        self._embeddings = embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._load_embeddings_model()
        vectors = model.embed_documents(texts)

        normalized_vectors: list[list[float]] = []
        for vector in vectors:
            if hasattr(vector, "tolist"):
                normalized_vectors.append(vector.tolist())
            else:
                normalized_vectors.append(list(vector))
        return normalized_vectors

    def _load_embeddings_model(self) -> Any:
        if self._embeddings is not None:
            return self._embeddings

        embeddings_class = self._resolve_hf_embeddings_class()
        model_kwargs: dict[str, Any] = {}

        resolved_token = self._token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if resolved_token:
            model_kwargs["token"] = resolved_token

        self._embeddings = embeddings_class(
            model_name=self._model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        return self._embeddings

    @staticmethod
    def _resolve_hf_embeddings_class() -> Any:
        try:
            module = importlib.import_module("langchain_huggingface")
            return module.HuggingFaceEmbeddings
        except ImportError:
            module = importlib.import_module("langchain_community.embeddings")
            return module.HuggingFaceEmbeddings


class QdrantVectorStore:
    """Thin Qdrant adapter for upserting CST embeddings."""

    def __init__(
        self,
        collection_name: str = "code_to_doc_cst",
        url: str | None = None,
        qdrant_api_key: str | None = None,
        client: Any | None = None,
        models_module: Any | None = None,
    ) -> None:
        self.collection_name = collection_name
        self._url = url
        self._qdrant_api_key = qdrant_api_key
        self._client = client
        self._models_module = models_module
        self._collection_ready = False

    def upsert_documents(self, documents: list[CSTDocument], embeddings: list[list[float]]) -> list[str]:
        if not documents:
            return []
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length.")

        self._ensure_collection(vector_size=len(embeddings[0]))
        client = self._get_client()
        models = self._get_models_module()
        ids = [document.doc_id for document in documents]
        points = [
            models.PointStruct(
                id=document.doc_id,
                vector=embedding,
                payload={**document.metadata, "cst": document.cst},
            )
            for document, embedding in zip(documents, embeddings, strict=True)
        ]

        client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        return ids

    def repository_exists(self, repository: str, ref: str | None = None) -> bool:
        if not self._collection_exists():
            return False

        client = self._get_client()
        scroll_kwargs = {
            "collection_name": self.collection_name,
            "scroll_filter": self._build_repository_filter(repository, ref),
            "limit": 1,
            "with_payload": False,
            "with_vectors": False,
        }
        try:
            scroll_result = client.scroll(**scroll_kwargs)
        except TypeError:
            scroll_kwargs.pop("with_vectors", None)
            scroll_result = client.scroll(**scroll_kwargs)
        points = self._normalize_scroll_points(scroll_result)
        return bool(points)

    def search_repository_documents(
        self,
        repository: str,
        ref: str | None,
        query_vector: list[float],
        limit: int = 8,
    ) -> list[RetrievedCSTDocument]:
        if not query_vector or not self._collection_exists():
            return []

        client = self._get_client()
        query_filter = self._build_repository_filter(repository, ref)

        if hasattr(client, "search"):
            search_kwargs = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "query_filter": query_filter,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False,
            }
            try:
                result = client.search(**search_kwargs)
            except TypeError:
                search_kwargs.pop("with_vectors", None)
                result = client.search(**search_kwargs)
            points = list(result)
        else:
            query_kwargs = {
                "collection_name": self.collection_name,
                "query": query_vector,
                "query_filter": query_filter,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False,
            }
            try:
                result = client.query_points(**query_kwargs)
            except TypeError:
                query_kwargs.pop("with_vectors", None)
                result = client.query_points(**query_kwargs)
            points = list(getattr(result, "points", result))

        return [self._to_retrieved_document(point) for point in points if getattr(point, "payload", None)]

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return

        client = self._get_client()
        exists = self._collection_exists()

        if not exists:
            model = self._get_models_module()
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=model.VectorParams(
                    size=vector_size,
                    distance=model.Distance.COSINE,
                ),
            )
            client.create_payload_index(
                collection_name=self.collection_name,
                field_name="repository",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        
            client.create_payload_index(
                collection_name=self.collection_name,
                field_name="ref",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        self._collection_ready = True

    def _collection_exists(self) -> bool:
        client = self._get_client()
        collection_exists = getattr(client, "collection_exists", None)
        if callable(collection_exists):
            return bool(collection_exists(self.collection_name))

        get_collections = getattr(client, "get_collections", None)
        if not callable(get_collections):
            return False

        response = get_collections()
        collections = getattr(response, "collections", response)
        for collection in collections or []:
            name = getattr(collection, "name", None)
            if name is None and isinstance(collection, dict):
                name = collection.get("name")
            if name == self.collection_name:
                return True
        return False

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self._url:
            raise ValueError("QDRANT_URL must be configured to store embeddings in Qdrant.")

        qdrant_client = importlib.import_module("qdrant_client")
        self._client = qdrant_client.QdrantClient(
            url=self._url,
            api_key=self._qdrant_api_key,
        )
        return self._client

    def _get_models_module(self) -> Any:
        if self._models_module is not None:
            return self._models_module

        self._models_module = importlib.import_module("qdrant_client.models")
        return self._models_module

    def _build_repository_filter(self, repository: str, ref: str | None) -> Any:
        models = self._get_models_module()
        ref_value = ref or "HEAD"
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="repository",
                    match=models.MatchValue(value=repository),
                ),
                models.FieldCondition(
                    key="ref",
                    match=models.MatchValue(value=ref_value),
                ),
            ]
        )

    @staticmethod
    def _normalize_scroll_points(scroll_result: Any) -> list[Any]:
        if isinstance(scroll_result, tuple):
            return list(scroll_result[0])

        points = getattr(scroll_result, "points", scroll_result)
        return list(points)

    @staticmethod
    def _to_retrieved_document(point: Any) -> RetrievedCSTDocument:
        payload = dict(getattr(point, "payload", {}) or {})
        metadata = {key: value for key, value in payload.items() if key != "cst"}
        return RetrievedCSTDocument(
            doc_id=str(getattr(point, "id", "")),
            path=str(payload.get("path", "")),
            language=str(payload.get("language", "unknown")),
            cst=str(payload.get("cst", "")),
            metadata=metadata,
            score=float(getattr(point, "score", 0.0) or 0.0),
        )


class RepositoryCSTEmbeddingIndexer:
    """LangGraph-driven CST embedding workflow."""

    def __init__(
        self,
        extractor: GitHubRepositoryExtractor | None = None,
        code_parser: TreeSitterCodeParser | None = None,
        embedder: LangChainEmbeddingModel | None = None,
        vector_store: QdrantVectorStore | None = None,
        workflow: Any | None = None,
    ) -> None:
        self._extractor = extractor or GitHubRepositoryExtractor()
        self._code_parser = code_parser or TreeSitterCodeParser()
        self._embedder = embedder or LangChainEmbeddingModel()
        self._vector_store = vector_store or QdrantVectorStore()
        self._workflow = workflow or self._build_workflow()

    def index_repository(self, repository_url: str, ref: str | None = None) -> RepositoryEmbeddingResult:
        initial_state: EmbeddingWorkflowState = {
            "repository_url": repository_url,
            "ref": ref,
        }
        final_state = self._workflow.invoke(initial_state)

        files = list(final_state.get("files", []))
        parsed_files = list(final_state.get("parsed_files", []))
        skipped_files = list(final_state.get("skipped_files", []))
        stored_ids = list(final_state.get("stored_ids", []))

        return RepositoryEmbeddingResult(
            repository=final_state["repository_name"],
            ref=final_state.get("ref"),
            fetched_files=len(files),
            parsed_files=parsed_files,
            skipped_files=skipped_files,
            stored_ids=stored_ids,
            collection_name=self._vector_store.collection_name,
        )

    def _build_workflow(self) -> Any:
        langgraph_graph = importlib.import_module("langgraph.graph")
        state_graph = langgraph_graph.StateGraph(EmbeddingWorkflowState)

        state_graph.add_node("fetch_repository", self._fetch_repository_node)
        state_graph.add_node("parse_repository_files", self._parse_repository_files_node)
        state_graph.add_node("embed_documents", self._embed_documents_node)
        state_graph.add_node("store_embeddings", self._store_embeddings_node)

        state_graph.set_entry_point("fetch_repository")
        state_graph.add_edge("fetch_repository", "parse_repository_files")
        state_graph.add_edge("parse_repository_files", "embed_documents")
        state_graph.add_edge("embed_documents", "store_embeddings")
        state_graph.add_edge("store_embeddings", langgraph_graph.END)

        return state_graph.compile()

    def _fetch_repository_node(self, state: EmbeddingWorkflowState) -> EmbeddingWorkflowState:
        repository_name, files = self._extractor.extract_repository_files(
            state["repository_url"],
            ref=state.get("ref"),
        )
        return {
            "repository_name": repository_name,
            "files": files,
        }

    def _parse_repository_files_node(self, state: EmbeddingWorkflowState) -> EmbeddingWorkflowState:
        parsed_files: list[ParsedFile] = []
        skipped_files: list[str] = []
        documents: list[CSTDocument] = []

        repository_name = state["repository_name"]
        ref = state.get("ref")

        for repository_file in state.get("files", []):
            parsed = self._code_parser.parse_file(repository_file)
            if parsed is None:
                skipped_files.append(repository_file.path)
                continue

            parsed_files.append(parsed)
            documents.append(self._to_document(repository_name, ref, repository_file, parsed))

        return {
            "parsed_files": parsed_files,
            "skipped_files": skipped_files,
            "documents": documents,
        }

    def _embed_documents_node(self, state: EmbeddingWorkflowState) -> EmbeddingWorkflowState:
        documents = list(state.get("documents", []))
        embeddings = self._embedder.embed_documents([document.cst for document in documents])
        return {"embeddings": embeddings}

    def _store_embeddings_node(self, state: EmbeddingWorkflowState) -> EmbeddingWorkflowState:
        documents = list(state.get("documents", []))
        embeddings = list(state.get("embeddings", []))
        stored_ids = self._vector_store.upsert_documents(documents, embeddings)
        return {"stored_ids": stored_ids}

    @staticmethod
    def _to_document(
        repository_name: str,
        ref: str | None,
        repository_file: RepositoryFile,
        parsed: ParsedFile,
    ) -> CSTDocument:
        ref_value = ref or "HEAD"
        identity = f"{repository_name}:{ref_value}:{repository_file.path}:{parsed.cst}"
        doc_id = sha256(identity.encode("utf-8")).hexdigest()
        valid_doc_id = str(uuid.UUID(doc_id[:32]))

        metadata = {
            "repository": repository_name,
            "ref": ref_value,
            "path": repository_file.path,
            "language": parsed.language,
            "root_type": parsed.root_type,
            "node_count": parsed.node_count,
            "has_error": parsed.has_error,
            "file_size": repository_file.size,
        }

        return CSTDocument(
            doc_id=valid_doc_id,
            path=repository_file.path,
            language=parsed.language,
            cst=parsed.cst,
            metadata=metadata,
        )
