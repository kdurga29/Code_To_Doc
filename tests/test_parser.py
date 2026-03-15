from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import importlib
import sys
import os

import pytest

from embeddings import (
    LangChainEmbeddingModel,
    CSTDocument,
    QdrantVectorStore,
    EmbeddingRuntimeConfig,
    RepositoryCSTEmbeddingIndexer,
)
from parser import (
    GitHubRepositoryExtractor,
    ParsedFile,
    RepositoryCodebaseParser,
    RepositoryFile,
    TreeSitterCSTSerializer,
    TreeSitterCodeParser,
)


@dataclass
class FakeContent:
    path: str
    type: str
    _decoded_content: bytes = b""
    size: int = 0

    @property
    def decoded_content(self) -> bytes:
        return self._decoded_content


class UnreadableContent(FakeContent):
    @property
    def decoded_content(self) -> bytes:
        raise RuntimeError("Cannot decode")


@dataclass
class FakeNode:
    type: str
    children: list["FakeNode"]
    start_byte: int = 0
    end_byte: int = 0


class TestGitHubRepositoryExtractor:
    def test_parse_repo_url_accepts_https_and_dot_git(self) -> None:
        parsed = GitHubRepositoryExtractor.parse_repo_url("https://github.com/octocat/Hello-World.git")
        assert parsed == "octocat/Hello-World"

    def test_parse_repo_url_accepts_owner_repo(self) -> None:
        parsed = GitHubRepositoryExtractor.parse_repo_url("octocat/Hello-World")
        assert parsed == "octocat/Hello-World"

    def test_parse_repo_url_accepts_host_prefix_without_scheme(self) -> None:
        parsed = GitHubRepositoryExtractor.parse_repo_url("github.com/octocat/Hello-World")
        assert parsed == "octocat/Hello-World"

    def test_parse_repo_url_rejects_non_github_host(self) -> None:
        with pytest.raises(ValueError):
            GitHubRepositoryExtractor.parse_repo_url("https://gitlab.com/octocat/Hello-World")

    def test_extract_repository_files_recurses_and_skips_excluded_dirs(self) -> None:
        repo = Mock()

        readme = FakeContent(path="README.md", type="file", _decoded_content=b"# docs", size=6)
        src_dir = FakeContent(path="src", type="dir")
        git_dir = FakeContent(path=".git", type="dir")
        app_py = FakeContent(path="src/app.py", type="file", _decoded_content=b"print('ok')", size=11)
        util_js = FakeContent(path="src/util.js", type="file", _decoded_content=b"console.log('ok')", size=17)

        def get_contents(path: str, ref: str | None = None):
            mapping = {
                "": [src_dir, readme, git_dir],
                "src": [app_py, util_js],
            }
            return mapping[path]

        repo.get_contents.side_effect = get_contents

        github_client = Mock()
        github_client.get_repo.return_value = repo

        extractor = GitHubRepositoryExtractor(github_client=github_client)
        full_name, files = extractor.extract_repository_files("https://github.com/acme/sample")

        assert full_name == "acme/sample"
        assert {file.path for file in files} == {"README.md", "src/app.py", "src/util.js"}

        requested_paths = [call.args[0] for call in repo.get_contents.call_args_list]
        assert ".git" not in requested_paths

    def test_extract_repository_files_skips_unreadable_file(self) -> None:
        repo = Mock()
        unreadable = UnreadableContent(path="bin/data.py", type="file")
        repo.get_contents.return_value = [unreadable]

        github_client = Mock()
        github_client.get_repo.return_value = repo

        extractor = GitHubRepositoryExtractor(github_client=github_client)
        _, files = extractor.extract_repository_files("https://github.com/acme/sample")

        assert files == []


class TestTreeSitterCSTSerializer:
    def test_serialize_generates_structured_cst_with_leaf_text(self) -> None:
        source = b"x=1"
        identifier = FakeNode(type="identifier", children=[], start_byte=0, end_byte=1)
        number = FakeNode(type="integer", children=[], start_byte=2, end_byte=3)
        assignment = FakeNode(type="assignment", children=[identifier, number])
        module = FakeNode(type="module", children=[assignment])

        serializer = TreeSitterCSTSerializer()
        result = serializer.serialize(module, source)

        assert result == '(module (assignment (identifier "x") (integer "1")))'

    def test_serialize_truncates_long_output(self) -> None:
        leaf = FakeNode(type="identifier", children=[], start_byte=0, end_byte=1)
        root = FakeNode(type="module", children=[leaf])

        serializer = TreeSitterCSTSerializer(max_cst_length=10)
        result = serializer.serialize(root, b"a")

        assert result.endswith("...")
        assert len(result) == 10


class TestTreeSitterCodeParser:
    def test_parse_file_returns_none_for_unsupported_extension(self) -> None:
        parser = TreeSitterCodeParser()
        parsed = parser.parse_file(RepositoryFile(path="README.md", content=b"# readme", size=8))
        assert parsed is None

    def test_parse_file_uses_language_module_and_returns_summary(self) -> None:
        file_to_parse = RepositoryFile(path="src/main.py", content=b"print('hello')", size=14)

        mock_root = Mock()
        mock_root.type = "module"
        mock_root.has_error = False
        mock_root.children = []

        mock_tree = Mock()
        mock_tree.root_node = mock_root

        mock_parser_instance = Mock()
        mock_parser_instance.parse.return_value = mock_tree

        mock_language_module = Mock()
        mock_language_module.language.return_value = "language-capsule"

        serializer = Mock()
        serializer.serialize.return_value = "(module)"

        parser_module = sys.modules[TreeSitterCodeParser.__module__]

        with patch.object(importlib, "import_module", return_value=mock_language_module) as import_module_mock, patch.object(
            parser_module,
            "Language",
            side_effect=lambda value: f"wrapped:{value}",
        ), patch.object(parser_module, "Parser", return_value=mock_parser_instance):
            parser = TreeSitterCodeParser(cst_serializer=serializer)
            parsed = parser.parse_file(file_to_parse)

        import_module_mock.assert_called_once_with("tree_sitter_python")
        mock_parser_instance.parse.assert_called_once_with(file_to_parse.content)
        serializer.serialize.assert_called_once_with(mock_root, file_to_parse.content)

        assert parsed == ParsedFile(
            path="src/main.py",
            language="python",
            root_type="module",
            node_count=1,
            has_error=False,
            cst="(module)",
        )

    def test_parser_instance_is_cached_per_extension(self) -> None:
        file_1 = RepositoryFile(path="a.py", content=b"x=1", size=3)
        file_2 = RepositoryFile(path="b.py", content=b"y=2", size=3)

        root = Mock()
        root.type = "module"
        root.has_error = False
        root.children = []

        tree = Mock()
        tree.root_node = root

        parser_instance = Mock()
        parser_instance.parse.return_value = tree

        language_module = Mock()
        language_module.language.return_value = "capsule"

        parser_module = sys.modules[TreeSitterCodeParser.__module__]

        with patch.object(importlib, "import_module", return_value=language_module) as import_module_mock, patch.object(
            parser_module,
            "Language",
            side_effect=lambda value: value,
        ), patch.object(parser_module, "Parser", return_value=parser_instance) as parser_ctor_mock:
            parser = TreeSitterCodeParser()
            parser.parse_file(file_1)
            parser.parse_file(file_2)

        assert parser_ctor_mock.call_count == 1
        assert import_module_mock.call_count == 1


class TestRepositoryCodebaseParser:
    def test_parse_repository_groups_parsed_and_skipped_files(self) -> None:
        extractor = Mock()
        extractor.extract_repository_files.return_value = (
            "acme/repo",
            [
                RepositoryFile(path="src/a.py", content=b"a=1", size=3),
                RepositoryFile(path="docs/readme.md", content=b"# hi", size=4),
            ],
        )

        code_parser = Mock()
        code_parser.parse_file.side_effect = [
            ParsedFile(
                path="src/a.py",
                language="python",
                root_type="module",
                node_count=1,
                has_error=False,
                cst="(module)",
            ),
            None,
        ]

        pipeline = RepositoryCodebaseParser(extractor=extractor, code_parser=code_parser)
        result = pipeline.parse_repository("https://github.com/acme/repo", ref="main")

        assert result.repository == "acme/repo"
        assert result.ref == "main"
        assert result.fetched_files == 2
        assert len(result.parsed_files) == 1
        assert result.parsed_files[0].path == "src/a.py"
        assert result.skipped_files == ["docs/readme.md"]


class TestEmbeddingRuntimeConfig:
    def test_from_env_reads_model_and_keys(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "EMBEDDING_MODEL_NAME": "BAAI/bge-base-en-v1.5",
                "HF_TOKEN": "hf-test-token",
                "QDRANT_COLLECTION_NAME": "repo-cst",
                "QDRANT_URL": "https://qdrant.example.com",
                "QDRANT_API_KEY": "qdrant-secret",
            },
            clear=True,
        ):
            config = EmbeddingRuntimeConfig.from_env()

        assert config.model_name == "BAAI/bge-base-en-v1.5"
        assert config.hf_token == "hf-test-token"
        assert config.qdrant_collection_name == "repo-cst"
        assert config.qdrant_url == "https://qdrant.example.com"
        assert config.qdrant_api_key == "qdrant-secret"


class TestLangChainEmbeddingModel:
    def test_embed_documents_uses_langchain_hf_embeddings_with_token(self) -> None:
        fake_backend = Mock()
        fake_backend.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        hf_embeddings_ctor = Mock(return_value=fake_backend)
        fake_module = SimpleNamespace(HuggingFaceEmbeddings=hf_embeddings_ctor)

        with patch("embeddings.importlib.import_module", return_value=fake_module), patch.dict(
            os.environ,
            {"HF_TOKEN": "hf-from-env"},
            clear=True,
        ):
            embedder = LangChainEmbeddingModel(model_name="BAAI/bge-base-en-v1.5")
            vectors = embedder.embed_documents(["first", "second"])

        hf_embeddings_ctor.assert_called_once_with(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"token": "hf-from-env"},
            encode_kwargs={"normalize_embeddings": True},
        )
        fake_backend.embed_documents.assert_called_once_with(["first", "second"])
        assert vectors == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_documents_falls_back_to_langchain_community(self) -> None:
        fake_backend = Mock()
        fake_backend.embed_documents.return_value = [[0.6, 0.7]]

        fallback_ctor = Mock(return_value=fake_backend)
        fallback_module = SimpleNamespace(HuggingFaceEmbeddings=fallback_ctor)

        def import_side_effect(module_name: str):
            if module_name == "langchain_huggingface":
                raise ImportError("not installed")
            if module_name == "langchain_community.embeddings":
                return fallback_module
            raise AssertionError(f"Unexpected module: {module_name}")

        with patch("embeddings.importlib.import_module", side_effect=import_side_effect):
            embedder = LangChainEmbeddingModel(model_name="BAAI/bge-base-en-v1.5")
            vectors = embedder.embed_documents(["only"])

        fallback_ctor.assert_called_once()
        fake_backend.embed_documents.assert_called_once_with(["only"])
        assert vectors == [[0.6, 0.7]]

    def test_embed_documents_returns_empty_for_empty_input(self) -> None:
        embedder = LangChainEmbeddingModel(embeddings=Mock())
        assert embedder.embed_documents([]) == []


class TestQdrantVectorStore:
    def test_upsert_documents_creates_collection_and_upserts_points(self) -> None:
        fake_client = Mock()
        fake_client.collection_exists.return_value = False

        vector_params = Mock(return_value={"size": 2, "distance": "cosine"})
        point_struct = Mock(side_effect=lambda id, vector, payload: {"id": id, "vector": vector, "payload": payload})
        fake_models = SimpleNamespace(
            Distance=SimpleNamespace(COSINE="cosine"),
            VectorParams=vector_params,
            PointStruct=point_struct,
        )

        store = QdrantVectorStore(
            collection_name="repo-cst",
            client=fake_client,
            models_module=fake_models,
        )

        documents = [
            CSTDocument(
                doc_id="doc-1",
                path="src/main.py",
                language="python",
                cst="(module)",
                metadata={"path": "src/main.py"},
            )
        ]
        ids = store.upsert_documents(documents, [[0.1, 0.2]])

        fake_client.collection_exists.assert_called_once_with("repo-cst")
        vector_params.assert_called_once_with(size=2, distance="cosine")
        fake_client.create_collection.assert_called_once_with(
            collection_name="repo-cst",
            vectors_config={"size": 2, "distance": "cosine"},
        )
        fake_client.upsert.assert_called_once_with(
            collection_name="repo-cst",
            points=[
                {
                    "id": "doc-1",
                    "vector": [0.1, 0.2],
                    "payload": {"path": "src/main.py", "cst": "(module)"},
                }
            ],
            wait=True,
        )
        assert ids == ["doc-1"]

    def test_upsert_documents_validates_lengths(self) -> None:
        store = QdrantVectorStore(client=Mock())
        documents = [
            CSTDocument(
                doc_id="doc-1",
                path="src/main.py",
                language="python",
                cst="(module)",
                metadata={},
            )
        ]

        with pytest.raises(ValueError):
            store.upsert_documents(documents, [])


class TestRepositoryCSTEmbeddingIndexer:
    def test_index_repository_parses_embeds_and_stores(self) -> None:
        extractor = Mock()
        extractor.extract_repository_files.return_value = (
            "acme/repo",
            [
                RepositoryFile(path="src/main.py", content=b"print('ok')", size=11),
                RepositoryFile(path="README.md", content=b"# docs", size=6),
            ],
        )

        code_parser = Mock()
        code_parser.parse_file.side_effect = [
            ParsedFile(
                path="src/main.py",
                language="python",
                root_type="module",
                node_count=4,
                has_error=False,
                cst="(module (expression_statement))",
            ),
            None,
        ]

        embedder = Mock()
        embedder.embed_documents.return_value = [[0.11, 0.22, 0.33]]

        vector_store = Mock()
        vector_store.collection_name = "repo-cst"
        vector_store.upsert_documents.return_value = ["stored-1"]

        class InlineWorkflow:
            def __init__(self, graph_indexer: RepositoryCSTEmbeddingIndexer) -> None:
                self._graph_indexer = graph_indexer

            def invoke(self, state: dict[str, object]) -> dict[str, object]:
                state = dict(state)
                state.update(self._graph_indexer._fetch_repository_node(state))
                state.update(self._graph_indexer._parse_repository_files_node(state))
                state.update(self._graph_indexer._embed_documents_node(state))
                state.update(self._graph_indexer._store_embeddings_node(state))
                return state

        indexer = RepositoryCSTEmbeddingIndexer(
            extractor=extractor,
            code_parser=code_parser,
            embedder=embedder,
            vector_store=vector_store,
            workflow=Mock(),
        )
        indexer._workflow = InlineWorkflow(indexer)

        result = indexer.index_repository("https://github.com/acme/repo", ref="main")

        embedder.embed_documents.assert_called_once_with(["(module (expression_statement))"])
        vector_store.upsert_documents.assert_called_once()

        docs_arg, embeddings_arg = vector_store.upsert_documents.call_args.args
        assert len(docs_arg) == 1
        assert docs_arg[0].path == "src/main.py"
        assert docs_arg[0].metadata["repository"] == "acme/repo"
        assert docs_arg[0].metadata["ref"] == "main"
        assert embeddings_arg == [[0.11, 0.22, 0.33]]

        assert result.repository == "acme/repo"
        assert result.ref == "main"
        assert result.fetched_files == 2
        assert result.skipped_files == ["README.md"]
        assert result.stored_ids == ["stored-1"]
        assert result.collection_name == "repo-cst"



