"""Repository extraction, Tree-sitter parsing, and CST embedding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
import importlib
import os
from dotenv import load_dotenv

load_dotenv()

from github import Github
from tree_sitter import Language, Parser


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".github",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "package-lock.json",
}


@dataclass(frozen=True)
class RepositoryFile:
    """A source file extracted from a repository."""

    path: str
    content: bytes
    size: int


@dataclass(frozen=True)
class ParsedFile:
    """Tree-sitter parse summary for one file."""

    path: str
    language: str
    root_type: str
    node_count: int
    has_error: bool
    cst: str


@dataclass(frozen=True)
class RepositoryParseResult:
    """Result of fetching and parsing a repository."""

    repository: str
    ref: str | None
    fetched_files: int
    parsed_files: list[ParsedFile]
    skipped_files: list[str]


class GitHubRepositoryExtractor:
    """Extract repository files from a GitHub URL using PyGithub."""

    def __init__(
        self,
        token: str | None = None,
        github_client: Any | None = None,
        excluded_dirs: set[str] | None = None,
    ) -> None:
        self._github = github_client or (Github(token) if token else Github())
        self._excluded_dirs = excluded_dirs or DEFAULT_EXCLUDED_DIRS

    @staticmethod
    def parse_repo_url(repository_url: str) -> str:
        """Convert a repository URL into `owner/repo` format."""
        cleaned = repository_url.strip()

        if "://" not in cleaned:
            direct_parts = [part for part in cleaned.strip("/").split("/") if part]
            if len(direct_parts) == 2:
                return f"{direct_parts[0]}/{direct_parts[1]}"
            if len(direct_parts) >= 3 and direct_parts[0].lower() in {"github.com", "www.github.com"}:
                repo_name = direct_parts[2]
                if repo_name.endswith(".git"):
                    repo_name = repo_name[:-4]
                return f"{direct_parts[1]}/{repo_name}"

        parsed = urlparse(cleaned)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme in {repository_url!r}")
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            raise ValueError(f"Unsupported host in {repository_url!r}")

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"Repository path not found in {repository_url!r}")

        owner = parts[0]
        repo = parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]

        if not owner or not repo:
            raise ValueError(f"Invalid repository URL: {repository_url!r}")

        return f"{owner}/{repo}"

    def extract_repository_files(
        self,
        repository_url: str,
        ref: str | None = None,
    ) -> tuple[str, list[RepositoryFile]]:
        """Fetch all repository files recursively."""
        full_name = self.parse_repo_url(repository_url)
        repo = self._github.get_repo(full_name)

        kwargs = {"ref": ref} if ref is not None else {}

        files: list[RepositoryFile] = []
        to_visit = list(self._as_list(repo.get_contents("", **kwargs)))

        while to_visit:
            content = to_visit.pop(0)
            if content.type == "dir":
                if self._should_skip_directory(content.path):
                    continue

                children = repo.get_contents(content.path, **kwargs)
                to_visit.extend(self._as_list(children))
                continue

            if content.type != "file":
                continue

            try:
                raw_content = content.decoded_content
            except Exception:
                continue

            files.append(
                RepositoryFile(
                    path=content.path,
                    content=raw_content,
                    size=getattr(content, "size", len(raw_content)),
                )
            )

        return full_name, files

    @staticmethod
    def _as_list(contents: Any) -> list[Any]:
        if isinstance(contents, list):
            return contents
        return [contents]

    def _should_skip_directory(self, path: str) -> bool:
        return any(part in self._excluded_dirs for part in path.split("/"))


class TreeSitterCSTSerializer:
    """Serialize Tree-sitter nodes into a deterministic CST string."""

    def __init__(
        self,
        max_leaf_text_length: int = 64,
        max_cst_length: int = 12_000,
    ) -> None:
        self._max_leaf_text_length = max_leaf_text_length
        self._max_cst_length = max_cst_length

    def serialize(self, root_node: Any, source_bytes: bytes) -> str:
        parts: list[str] = []
        stack: list[tuple[Any, bool, int]] = [(root_node, False, 0)]

        while stack:
            node, opened, child_index = stack.pop()
            children = list(getattr(node, "children", []) or [])

            if not opened:
                node_type = getattr(node, "type", "unknown")
                parts.append(f"({node_type}")
                if not children:
                    leaf_text = self._leaf_text(node, source_bytes)
                    if leaf_text:
                        parts.append(f' "{leaf_text}"')
                    parts.append(")")
                    continue

                stack.append((node, True, 0))
                continue

            if child_index >= len(children):
                parts.append(")")
                continue

            parts.append(" ")
            stack.append((node, True, child_index + 1))
            stack.append((children[child_index], False, 0))

        serialized = "".join(parts)
        if len(serialized) > self._max_cst_length:
            return f"{serialized[: self._max_cst_length - 3]}..."
        return serialized

    def _leaf_text(self, node: Any, source_bytes: bytes) -> str | None:
        start_byte = getattr(node, "start_byte", None)
        end_byte = getattr(node, "end_byte", None)
        if not isinstance(start_byte, int) or not isinstance(end_byte, int):
            return None
        if start_byte < 0 or end_byte < 0 or end_byte <= start_byte:
            return None
        if end_byte > len(source_bytes):
            return None

        snippet = source_bytes[start_byte:end_byte].decode("utf-8", errors="ignore")
        cleaned = " ".join(snippet.split()).strip()
        if not cleaned:
            return None

        cleaned = cleaned.replace('"', '\\"')
        if len(cleaned) > self._max_leaf_text_length:
            cleaned = f"{cleaned[: self._max_leaf_text_length - 3]}..."
        return cleaned


class TreeSitterCodeParser:
    """Parse repository files with Tree-sitter."""

    DEFAULT_LANGUAGE_REGISTRY = {
        ".py": ("python", "tree_sitter_python"),
        ".js": ("javascript", "tree_sitter_javascript"),
        ".jsx": ("javascript", "tree_sitter_javascript"),
        ".mjs": ("javascript", "tree_sitter_javascript"),
        ".cjs": ("javascript", "tree_sitter_javascript"),
    }

    def __init__(
        self,
        language_registry: dict[str, tuple[str, str]] | None = None,
        cst_serializer: TreeSitterCSTSerializer | None = None,
    ) -> None:
        self._language_registry = language_registry or self.DEFAULT_LANGUAGE_REGISTRY
        self._parser_cache: dict[str, tuple[Parser, str]] = {}
        self._cst_serializer = cst_serializer or TreeSitterCSTSerializer()

    def parse_file(self, repository_file: RepositoryFile) -> ParsedFile | None:
        """Parse one repository file if its extension is supported."""
        extension = os.path.splitext(repository_file.path)[1].lower()
        if extension not in self._language_registry:
            return None

        parser, language_name = self._get_parser_for_extension(extension)
        tree = parser.parse(repository_file.content)
        root_node = tree.root_node

        return ParsedFile(
            path=repository_file.path,
            language=language_name,
            root_type=root_node.type,
            node_count=self._count_nodes(root_node),
            has_error=bool(getattr(root_node, "has_error", False)),
            cst=self._cst_serializer.serialize(root_node, repository_file.content),
        )

    def _get_parser_for_extension(self, extension: str) -> tuple[Parser, str]:
        cached = self._parser_cache.get(extension)
        if cached:
            return cached

        language_name, module_name = self._language_registry[extension]
        module = importlib.import_module(module_name)

        language_builder = getattr(module, "language", None)
        if language_builder is None:
            raise ValueError(f"Module '{module_name}' does not expose language()")

        raw_language = language_builder()
        language = self._coerce_language(raw_language)

        parser = Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(language)
        else:
            parser.language = language

        resolved = (parser, language_name)
        self._parser_cache[extension] = resolved
        return resolved

    @staticmethod
    def _coerce_language(raw_language: Any) -> Any:
        if isinstance(Language, type) and isinstance(raw_language, Language):
            return raw_language

        try:
            return Language(raw_language)
        except TypeError:
            return raw_language

    @staticmethod
    def _count_nodes(root: Any) -> int:
        total = 0
        stack: list[Any] = [root]

        while stack:
            node = stack.pop()
            total += 1
            children = getattr(node, "children", [])
            for child in children:
                stack.append(child)

        return total


class RepositoryCodebaseParser:
    """Orchestrates extraction and parsing for a repository URL."""

    def __init__(
        self,
        extractor: GitHubRepositoryExtractor | None = None,
        code_parser: TreeSitterCodeParser | None = None,
    ) -> None:
        self._extractor = extractor or GitHubRepositoryExtractor()
        self._code_parser = code_parser or TreeSitterCodeParser()

    def parse_repository(self, repository_url: str, ref: str | None = None) -> RepositoryParseResult:
        repository_name, files = self._extractor.extract_repository_files(repository_url, ref=ref)

        parsed_files: list[ParsedFile] = []
        skipped_files: list[str] = []

        for repository_file in files:
            parsed = self._code_parser.parse_file(repository_file)
            if parsed is None:
                skipped_files.append(repository_file.path)
                continue
            parsed_files.append(parsed)

        return RepositoryParseResult(
            repository=repository_name,
            ref=ref,
            fetched_files=len(files),
            parsed_files=parsed_files,
            skipped_files=skipped_files,
        )



