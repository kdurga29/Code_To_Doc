"""RAG documentation generation on top of stored repository CST embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    from embeddings import (
        DEFAULT_DOCUMENTATION_MODEL,
        QdrantVectorStore,
        RepositoryEmbeddingResult,
        RetrievedCSTDocument,
    )
    from parser import GitHubRepositoryExtractor
except ImportError:
    from src.embeddings import (
        DEFAULT_DOCUMENTATION_MODEL,
        QdrantVectorStore,
        RepositoryEmbeddingResult,
        RetrievedCSTDocument,
    )
    from src.parser import GitHubRepositoryExtractor


DEFAULT_RETRIEVAL_QUERIES = [
    "Repository overview, purpose, and primary capabilities.",
    "Architecture, key modules, and important execution flow.",
    "Core classes, functions, and integration points developers should understand.",
    "Setup, configuration, dependencies, and operational considerations.",
]


@dataclass(frozen=True)
class RepositoryDocumentationResult:
    """Documentation produced from retrieved repository context."""

    repository: str
    ref: str | None
    markdown: str
    retrieved_documents: list[RetrievedCSTDocument]


@dataclass(frozen=True)
class RepositoryDocumentationWorkflowResult:
    """Top-level outcome for index-or-retrieve documentation generation."""

    repository: str
    ref: str | None
    already_indexed: bool
    indexed_now: bool
    index_result: RepositoryEmbeddingResult | None
    documentation: RepositoryDocumentationResult


def setup_local_hf_model(
    model_id: str = DEFAULT_DOCUMENTATION_MODEL,
    hf_token: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    repetition_penalty: float = 1.1,
    load_in_4bit: bool = True,
) -> Any:
    """Create a LangChain-wrapped local Hugging Face generation pipeline."""

    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")

    auth_kwargs: dict[str, Any] = {}
    resolved_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if resolved_token:
        auth_kwargs["token"] = resolved_token

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **auth_kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"device_map": "auto"}
    use_cuda = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())

    if use_cuda:
        model_kwargs["torch_dtype"] = torch.bfloat16
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True

    auto_model = transformers.AutoModelForCausalLM
    try:
        model = auto_model.from_pretrained(model_id, **auth_kwargs, **model_kwargs)
    except Exception as exc:
        if not use_cuda or "load_in_4bit" not in model_kwargs:
            raise ValueError(f"Unable to load local documentation model '{model_id}': {exc}") from exc

        fallback_model_kwargs = dict(model_kwargs)
        fallback_model_kwargs.pop("load_in_4bit", None)
        fallback_model_kwargs.pop("torch_dtype", None)
        try:
            model = auto_model.from_pretrained(model_id, **auth_kwargs, **fallback_model_kwargs)
        except Exception as fallback_exc:
            raise ValueError(
                "Unable to load the local documentation model. "
                "Tried 4-bit GPU loading first, then standard loading, and both failed."
            ) from fallback_exc

    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = 512
    model.generation_config.temperature = 0.1
    model.generation_config.do_sample = True
    model.generation_config.repetition_penalty = 1.1
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    generation_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    return _resolve_langchain_pipeline_class()(pipeline=generation_pipeline)


def _resolve_langchain_pipeline_class() -> Any:
    try:
        module = importlib.import_module("langchain_huggingface")
        return module.HuggingFacePipeline
    except (AttributeError, ImportError):
        module = importlib.import_module("langchain_community.llms")
        return module.HuggingFacePipeline


class HuggingFaceDocumentationGenerator:
    """Generate documentation markdown with a local Hugging Face model."""

    def __init__(
        self,
        model_name: str = DEFAULT_DOCUMENTATION_MODEL,
        hf_token: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        repetition_penalty: float = 1.1,
        load_in_4bit: bool = True,
        llm: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._hf_token = hf_token
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._load_in_4bit = load_in_4bit
        self._llm = llm

    def generate_markdown(
        self,
        repository: str,
        ref: str | None,
        retrieved_documents: list[RetrievedCSTDocument],
        context_char_limit: int = 16_000,
    ) -> str:
        if not retrieved_documents:
            raise ValueError("No retrieved repository context was available for documentation generation.")

        prompt = self._build_prompt(
            repository=repository,
            ref=ref,
            retrieved_documents=retrieved_documents,
            context_char_limit=context_char_limit,
        )
        response = self._load_llm().invoke(prompt)
        markdown = self._extract_markdown(response)
        if not markdown:
            raise ValueError("Local Hugging Face model returned no text for documentation generation.")
        return markdown

    def _build_prompt(
        self,
        repository: str,
        ref: str | None,
        retrieved_documents: list[RetrievedCSTDocument],
        context_char_limit: int,
    ) -> str:
        context_blocks: list[str] = []
        current_size = 0

        for document in retrieved_documents:
            metadata_lines = [
                f"path: {document.path}",
                f"language: {document.language}",
                f"score: {document.score:.4f}",
            ]
            for key in ("root_type", "node_count", "has_error", "file_size"):
                if key in document.metadata:
                    metadata_lines.append(f"{key}: {document.metadata[key]}")

            block = "\n".join(
                [
                    "### Retrieved File",
                    *metadata_lines,
                    "```text",
                    document.cst,
                    "```",
                ]
            )

            if current_size + len(block) > context_char_limit and context_blocks:
                break

            context_blocks.append(block)
            current_size += len(block)

        ref_value = ref or "HEAD"
        joined_context = "\n\n".join(context_blocks)
        return (
            "### Instruction:\n"
            "You are a senior software architect producing repository documentation in Markdown.\n"
            "Use only the retrieved context below.\n"
            "If something cannot be inferred confidently, say that it is inferred or unknown.\n"
            "Return Markdown only.\n\n"
            f"Repository: {repository}\n"
            f"Reference: {ref_value}\n\n"
            "Write documentation with these sections:\n"
            "1. Title and concise overview\n"
            "2. Architecture summary\n"
            "3. Key modules and responsibilities\n"
            "4. Execution or data flow\n"
            "5. Setup, configuration, and dependencies\n"
            "6. Operational notes, limitations, or gaps\n\n"
            "Retrieved repository context:\n\n"
            f"{joined_context}\n\n"
            "### Response:\n" 
        )

    def _load_llm(self) -> Any:
        if self._llm is None:
            self._llm = setup_local_hf_model(
                model_id=self._model_name,
                hf_token=self._hf_token,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                repetition_penalty=self._repetition_penalty,
                load_in_4bit=self._load_in_4bit,
            )
        return self._llm

    @staticmethod
    def _extract_markdown(response: Any) -> str:
        if isinstance(response, str):
            return response.strip()

        if isinstance(response, list):
            parts = [HuggingFaceDocumentationGenerator._extract_markdown(item) for item in response]
            return "\n".join(part for part in parts if part).strip()

        if isinstance(response, dict):
            for key in ("text", "generated_text", "content"):
                value = response.get(key)
                if value:
                    return HuggingFaceDocumentationGenerator._extract_markdown(value)

        content = getattr(response, "content", None)
        if content:
            return HuggingFaceDocumentationGenerator._extract_markdown(content)

        return ""


class RepositoryDocumentationRAGChain:
    """Retrieve stored CST context and ask an LLM to produce repository documentation."""

    def __init__(
        self,
        embedder: Any,
        vector_store: QdrantVectorStore,
        generator: HuggingFaceDocumentationGenerator,
        retrieval_queries: list[str] | None = None,
        retrieval_limit: int = 8,
        context_char_limit: int = 16_000,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._generator = generator
        self._retrieval_queries = retrieval_queries or DEFAULT_RETRIEVAL_QUERIES
        self._retrieval_limit = retrieval_limit
        self._context_char_limit = context_char_limit

    def generate(self, repository_url: str, ref: str | None = None) -> RepositoryDocumentationResult:
        repository = GitHubRepositoryExtractor.parse_repo_url(repository_url)
        if not self._vector_store.repository_exists(repository, ref):
            raise ValueError(f"Repository '{repository}' was not found in the vector store.")

        retrieved_by_id: dict[str, RetrievedCSTDocument] = {}
        for query in self._retrieval_queries:
            query_vector = self._embedder.embed_documents([query])[0]
            query_results = self._vector_store.search_repository_documents(
                repository=repository,
                ref=ref,
                query_vector=query_vector,
                limit=self._retrieval_limit,
            )

            for document in query_results:
                existing = retrieved_by_id.get(document.doc_id)
                if existing is None or document.score > existing.score:
                    retrieved_by_id[document.doc_id] = document

        retrieved_documents = sorted(
            retrieved_by_id.values(),
            key=lambda document: document.score,
            reverse=True,
        )[: self._retrieval_limit]

        if not retrieved_documents:
            raise ValueError(f"No relevant CST context could be retrieved for '{repository}'.")

        markdown = self._generator.generate_markdown(
            repository=repository,
            ref=ref,
            retrieved_documents=retrieved_documents,
            context_char_limit=self._context_char_limit,
        )
        return RepositoryDocumentationResult(
            repository=repository,
            ref=ref,
            markdown=markdown,
            retrieved_documents=retrieved_documents,
        )


class RepositoryDocumentationWorkflow:
    """Generate repository documentation, indexing first when required."""

    def __init__(
        self,
        indexer: Any,
        rag_chain: RepositoryDocumentationRAGChain,
        vector_store: QdrantVectorStore,
    ) -> None:
        self._indexer = indexer
        self._rag_chain = rag_chain
        self._vector_store = vector_store

    def run(self, repository_url: str, ref: str | None = None) -> RepositoryDocumentationWorkflowResult:
        repository = GitHubRepositoryExtractor.parse_repo_url(repository_url)
        already_indexed = self._vector_store.repository_exists(repository, ref)

        index_result: RepositoryEmbeddingResult | None = None
        if not already_indexed:
            index_result = self._indexer.index_repository(repository_url, ref=ref)
            if not index_result.stored_ids:
                raise ValueError(
                    f"Repository '{repository}' was indexed, but no supported source files were stored for retrieval."
                )

        documentation = self._rag_chain.generate(repository_url, ref=ref)
        return RepositoryDocumentationWorkflowResult(
            repository=repository,
            ref=ref,
            already_indexed=already_indexed,
            indexed_now=index_result is not None,
            index_result=index_result,
            documentation=documentation,
        )
