"""Helpers for publishing generated documentation to GitHub wiki pages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote
import os
import re
import subprocess


@dataclass(frozen=True)
class PublishedWikiPage:
    """Metadata describing a page committed to a repository wiki."""

    repository: str
    page_title: str
    page_path: str
    page_url: str


def build_wiki_page_markdown(
    repository_url: str,
    ref: str | None,
    markdown: str,
    after: str | None = None,
    generated_by: str = "Code To Doc",
) -> str:
    """Add a short generated-by header before repository documentation markdown."""

    header_lines = [
        f"> Generated automatically by {generated_by}.",
        f"> Source repository: [{repository_url}]({repository_url})",
        f"> Source ref: `{ref or 'HEAD'}`",
    ]
    if after:
        header_lines.append(f"> Source commit: `{after}`")

    generated_markdown = markdown.lstrip()
    return "\n".join([*header_lines, "", generated_markdown])


class GitHubWikiPublisher:
    """Publish markdown documents into a repository's GitHub wiki git repository."""

    def __init__(
        self,
        token: str | None = None,
        author_name: str | None = None,
        author_email: str | None = None,
        git_command: str = "git",
        temp_dir_factory: Callable[[], TemporaryDirectory] | None = None,
    ) -> None:
        self._token = token or os.getenv("GITHUB_WIKI_TOKEN") or os.getenv("GITHUB_TOKEN")
        self._author_name = author_name or os.getenv("GITHUB_WIKI_AUTHOR_NAME", "Code To Doc Bot")
        self._author_email = author_email or os.getenv("GITHUB_WIKI_AUTHOR_EMAIL", "code-to-doc-bot@example.com")
        self._git_command = git_command
        self._temp_dir_factory = temp_dir_factory or (lambda: TemporaryDirectory(prefix="code-to-doc-wiki-"))

    def publish_page(
        self,
        repository: str,
        page_title: str,
        markdown: str,
        commit_message: str,
    ) -> PublishedWikiPage:
        """Commit a new markdown page into the target repository wiki."""

        if not self._token:
            raise ValueError("A GitHub token is required to publish generated documentation to the wiki.")

        with self._temp_dir_factory() as temp_dir:
            wiki_dir = Path(temp_dir) / "wiki"
            self._run_git(["clone", self._build_clone_url(repository), str(wiki_dir)], cwd=Path(temp_dir))
            self._run_git(["config", "user.name", self._author_name], cwd=wiki_dir)
            self._run_git(["config", "user.email", self._author_email], cwd=wiki_dir)

            page_path = self._resolve_page_path(wiki_dir, page_title)
            page_path.write_text(markdown.strip() + "\n", encoding="utf-8")

            relative_page_path = page_path.relative_to(wiki_dir).as_posix()
            self._run_git(["add", relative_page_path], cwd=wiki_dir)
            self._run_git(["commit", "-m", commit_message], cwd=wiki_dir)
            self._run_git(["push", "origin", "HEAD"], cwd=wiki_dir)

            page_name = page_path.stem
            return PublishedWikiPage(
                repository=repository,
                page_title=page_name,
                page_path=relative_page_path,
                page_url=f"https://github.com/{repository}/wiki/{quote(page_name)}",
            )

    def _build_clone_url(self, repository: str) -> str:
        encoded_token = quote(self._token or "", safe="")
        return f"https://x-access-token:{encoded_token}@github.com/{repository}.wiki.git"

    def _resolve_page_path(self, wiki_dir: Path, page_title: str) -> Path:
        safe_title = self._sanitize_page_title(page_title)
        candidate = wiki_dir / f"{safe_title}.md"
        suffix = 2

        while candidate.exists():
            candidate = wiki_dir / f"{safe_title}-{suffix}.md"
            suffix += 1

        return candidate

    @staticmethod
    def _sanitize_page_title(page_title: str) -> str:
        cleaned = re.sub(r"[\\/:*?\"<>|#%^~]+", "-", page_title.strip())
        cleaned = re.sub(r"\s+", "-", cleaned)
        cleaned = re.sub(r"-{2,}", "-", cleaned)
        cleaned = cleaned.strip("-.")
        return cleaned or "Generated-Documentation"

    def _run_git(self, args: list[str], cwd: Path) -> None:
        completed = subprocess.run(
            [self._git_command, *args],
            cwd=str(cwd),
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or "git command failed"
            raise RuntimeError(f"Failed to publish wiki page via git {' '.join(args)}: {details}")
