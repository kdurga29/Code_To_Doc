"""FastAPI webhook endpoint that generates repository documentation and publishes it to GitHub wiki."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

try:
    from documentation import RepositoryDocumentationWorkflow
    from runtime import build_documentation_workflow
    from wiki import GitHubWikiPublisher, PublishedWikiPage, build_wiki_page_markdown
except ImportError:
    from src.documentation import RepositoryDocumentationWorkflow
    from src.runtime import build_documentation_workflow
    from src.wiki import GitHubWikiPublisher, PublishedWikiPage, build_wiki_page_markdown


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GitHubPushEventDetails:
    """Subset of push-event metadata needed by the documentation workflow."""

    repository: str
    repository_url: str
    ref: str | None
    after: str | None


@dataclass(frozen=True)
class WebhookProcessingResult:
    """Result of processing one GitHub push webhook."""

    repository: str
    ref: str | None
    page: PublishedWikiPage


class GitHubWebhookDocumentationService:
    """Turn GitHub push events into generated wiki documentation pages."""

    def __init__(
        self,
        workflow: RepositoryDocumentationWorkflow,
        wiki_publisher: GitHubWikiPublisher,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._workflow = workflow
        self._wiki_publisher = wiki_publisher
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    def parse_push_event(self, payload: Mapping[str, Any]) -> GitHubPushEventDetails:
        """Validate and normalize the repository information from a GitHub push event."""

        repository_data = payload.get("repository")
        if not isinstance(repository_data, Mapping):
            raise ValueError("GitHub webhook payload did not include a repository object.")

        repository = repository_data.get("full_name")
        if not isinstance(repository, str) or not repository.strip():
            raise ValueError("GitHub webhook payload did not include repository.full_name.")

        repository_url = repository_data.get("html_url")
        if not isinstance(repository_url, str) or not repository_url.strip():
            repository_url = f"https://github.com/{repository}"

        ref_value = payload.get("ref")
        ref = self._normalize_ref(ref_value if isinstance(ref_value, str) else None)
        after = payload.get("after")
        normalized_after = after if isinstance(after, str) and after.strip() else None

        return GitHubPushEventDetails(
            repository=repository,
            repository_url=repository_url,
            ref=ref,
            after=normalized_after,
        )

    def handle_push_event(self, payload: Mapping[str, Any]) -> WebhookProcessingResult:
        """Generate documentation for a push event and publish it as a wiki page."""

        details = self.parse_push_event(payload)
        workflow_result = self._workflow.run(details.repository_url, ref=details.ref)
        page_title = self._build_page_title(details)
        page_markdown = build_wiki_page_markdown(
            repository_url=details.repository_url,
            ref=details.ref,
            after=details.after,
            markdown=workflow_result.documentation.markdown,
            generated_by="Code To Doc webhook",
        )
        page = self._wiki_publisher.publish_page(
            repository=workflow_result.repository,
            page_title=page_title,
            markdown=page_markdown,
            commit_message=self._build_commit_message(details),
        )
        return WebhookProcessingResult(
            repository=workflow_result.repository,
            ref=details.ref,
            page=page,
        )

    def process_push_event_safely(self, payload: Mapping[str, Any]) -> None:
        """Background-task wrapper that logs failures instead of surfacing them to GitHub."""

        try:
            result = self.handle_push_event(payload)
            logger.info(
                "Published generated documentation page '%s' for %s (%s)",
                result.page.page_title,
                result.repository,
                result.ref or "HEAD",
            )
        except Exception:
            logger.exception("Failed to process GitHub webhook for documentation generation.")

    def _build_page_title(self, details: GitHubPushEventDetails) -> str:
        ref_label = details.ref or "HEAD"
        sha_label = (details.after or "unknown")[:7]
        timestamp = self._now_provider().strftime("%Y%m%d-%H%M%S")
        return f"Auto Docs {ref_label} {sha_label} {timestamp}"

    @staticmethod
    def _build_commit_message(details: GitHubPushEventDetails) -> str:
        ref_label = details.ref or "HEAD"
        sha_label = (details.after or "unknown")[:7]
        return f"Add generated documentation for {ref_label} ({sha_label})"

    @staticmethod
    def _normalize_ref(ref: str | None) -> str | None:
        if not ref:
            return None
        prefixes = ("refs/heads/", "refs/tags/")
        for prefix in prefixes:
            if ref.startswith(prefix):
                return ref[len(prefix) :]
        return ref


class GitHubWebhookSignatureVerifier:
    """Validate GitHub webhook HMAC signatures when a secret is configured."""

    def __init__(self, secret: str | None = None) -> None:
        self._secret = secret or os.getenv("GITHUB_WEBHOOK_SECRET")

    def verify(self, body: bytes, signature_header: str | None) -> bool:
        if not self._secret:
            return True
        if not signature_header or not signature_header.startswith("sha256="):
            return False

        expected = hmac.new(
            self._secret.encode("utf-8"),
            msg=body,
            digestmod=hashlib.sha256,
        ).hexdigest()
        provided = signature_header.split("=", maxsplit=1)[1]
        return hmac.compare_digest(expected, provided)


def build_webhook_service() -> GitHubWebhookDocumentationService:
    """Construct the end-to-end webhook processor from environment settings."""

    workflow = build_documentation_workflow(github_token=os.getenv("GITHUB_TOKEN"))
    wiki_publisher = GitHubWikiPublisher()
    return GitHubWebhookDocumentationService(
        workflow=workflow,
        wiki_publisher=wiki_publisher,
    )


def create_app(
    processor: GitHubWebhookDocumentationService | None = None,
    signature_verifier: GitHubWebhookSignatureVerifier | None = None,
) -> FastAPI:
    """Create the FastAPI application that receives GitHub webhooks."""

    app = FastAPI(title="Code To Doc Webhook API")
    app.state.processor = processor or build_webhook_service()
    app.state.signature_verifier = signature_verifier or GitHubWebhookSignatureVerifier()

    @app.post("/webhooks/github")
    async def github_webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        x_github_event: str = Header(..., alias="X-GitHub-Event"),
        x_hub_signature_256: str | None = Header(default=None, alias="X-Hub-Signature-256"),
    ) -> dict[str, Any]:
        body = await request.body()
        if not app.state.signature_verifier.verify(body, x_hub_signature_256):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid GitHub webhook signature.",
            )

        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Webhook payload must be valid JSON.",
            ) from exc

        if x_github_event == "ping":
            return {"status": "ok", "message": "GitHub webhook received."}

        if x_github_event != "push":
            return {"status": "ignored", "event": x_github_event}

        try:
            details = app.state.processor.parse_push_event(payload)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        background_tasks.add_task(app.state.processor.process_push_event_safely, payload)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "accepted",
                "repository": details.repository,
                "ref": details.ref or "HEAD",
            },
        )

    return app


app = create_app()
