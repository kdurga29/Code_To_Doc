from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone
from unittest.mock import Mock

from fastapi.testclient import TestClient

from documentation import RepositoryDocumentationResult, RepositoryDocumentationWorkflowResult
from webhook_api import (
    GitHubWebhookDocumentationService,
    GitHubWebhookSignatureVerifier,
    create_app,
)
from wiki import PublishedWikiPage, build_wiki_page_markdown


def test_signature_verifier_checks_sha256_signature() -> None:
    verifier = GitHubWebhookSignatureVerifier(secret="top-secret")
    body = b'{"zen":"keep it logically awesome"}'
    signature = hmac.new(b"top-secret", body, hashlib.sha256).hexdigest()

    assert verifier.verify(body, f"sha256={signature}") is True
    assert verifier.verify(body, "sha256=bad-signature") is False
    assert verifier.verify(body, None) is False


def test_documentation_service_generates_wiki_page_from_push_event() -> None:
    workflow = Mock()
    workflow.run.return_value = RepositoryDocumentationWorkflowResult(
        repository="acme/repo",
        ref="main",
        already_indexed=True,
        indexed_now=False,
        index_result=None,
        documentation=RepositoryDocumentationResult(
            repository="acme/repo",
            ref="main",
            markdown="# Generated Docs",
            retrieved_documents=[],
        ),
    )

    wiki_publisher = Mock()
    wiki_publisher.publish_page.return_value = PublishedWikiPage(
        repository="acme/repo",
        page_title="Auto-Docs-main-acde123-20260331-120000",
        page_path="Auto-Docs-main-acde123-20260331-120000.md",
        page_url="https://github.com/acme/repo/wiki/Auto-Docs-main-acde123-20260331-120000",
    )

    service = GitHubWebhookDocumentationService(
        workflow=workflow,
        wiki_publisher=wiki_publisher,
        now_provider=lambda: datetime(2026, 3, 31, 12, 0, 0, tzinfo=timezone.utc),
    )
    payload = {
        "ref": "refs/heads/main",
        "after": "acde12345f6",
        "repository": {
            "full_name": "acme/repo",
            "html_url": "https://github.com/acme/repo",
        },
    }

    result = service.handle_push_event(payload)

    assert result.repository == "acme/repo"
    assert result.ref == "main"
    workflow.run.assert_called_once_with("https://github.com/acme/repo", ref="main")
    wiki_publisher.publish_page.assert_called_once()
    publish_kwargs = wiki_publisher.publish_page.call_args.kwargs
    assert publish_kwargs["repository"] == "acme/repo"
    assert publish_kwargs["page_title"] == "Auto Docs main acde123 20260331-120000"
    assert publish_kwargs["commit_message"] == "Add generated documentation for main (acde123)"
    assert "> Source ref: `main`" in publish_kwargs["markdown"]
    assert "# Generated Docs" in publish_kwargs["markdown"]


def test_build_wiki_page_markdown_adds_source_header() -> None:
    markdown = build_wiki_page_markdown(
        repository_url="https://github.com/acme/repo",
        ref="main",
        after="abc123",
        markdown="# Docs",
        generated_by="Test Harness",
    )

    assert markdown.startswith("> Generated automatically by Test Harness.")
    assert "> Source repository: [https://github.com/acme/repo](https://github.com/acme/repo)" in markdown
    assert "> Source ref: `main`" in markdown
    assert "> Source commit: `abc123`" in markdown
    assert markdown.endswith("# Docs")


def test_create_app_handles_ping_event() -> None:
    processor = Mock()
    signature_verifier = Mock()
    signature_verifier.verify.return_value = True
    app = create_app(
        processor=processor,
        signature_verifier=signature_verifier,
    )
    client = TestClient(app)

    response = client.post(
        "/webhooks/github",
        headers={"X-GitHub-Event": "ping"},
        json={"zen": "keep it logically awesome"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    processor.parse_push_event.assert_not_called()


def test_create_app_rejects_invalid_signature() -> None:
    processor = Mock()
    app = create_app(
        processor=processor,
        signature_verifier=GitHubWebhookSignatureVerifier(secret="top-secret"),
    )
    client = TestClient(app)

    response = client.post(
        "/webhooks/github",
        headers={
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": "sha256=wrong",
        },
        json={"repository": {"full_name": "acme/repo"}},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid GitHub webhook signature."


def test_create_app_accepts_push_event_and_runs_background_processing() -> None:
    processor = Mock()
    processor.parse_push_event.return_value = Mock(repository="acme/repo", ref="main")
    signature_verifier = Mock()
    signature_verifier.verify.return_value = True

    app = create_app(
        processor=processor,
        signature_verifier=signature_verifier,
    )
    client = TestClient(app)
    payload = {
        "ref": "refs/heads/main",
        "after": "acde12345f6",
        "repository": {
            "full_name": "acme/repo",
            "html_url": "https://github.com/acme/repo",
        },
    }

    response = client.post(
        "/webhooks/github",
        headers={"X-GitHub-Event": "push"},
        json=payload,
    )

    assert response.status_code == 202
    assert response.json() == {
        "status": "accepted",
        "repository": "acme/repo",
        "ref": "main",
    }
    processor.parse_push_event.assert_called_once_with(payload)
    processor.process_push_event_safely.assert_called_once_with(payload)
