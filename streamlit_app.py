"""Streamlit UI for repository parsing + CST embeddings."""

from __future__ import annotations

from datetime import datetime
import os

import streamlit as st
from dotenv import load_dotenv

from src.embeddings import EmbeddingRuntimeConfig
from src.runtime import build_documentation_workflow
from src.wiki import GitHubWikiPublisher, build_wiki_page_markdown

load_dotenv()


def _build_webhook_url(public_base_url: str) -> str:
    normalized = public_base_url.strip().rstrip("/")
    if not normalized:
        return "/webhooks/github"
    return f"{normalized}/webhooks/github"


def _default_wiki_page_title(repository: str, ref: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ref_label = ref or "HEAD"
    return f"Manual Docs {repository} {ref_label} {timestamp}"


def _config_status_rows(runtime_config: EmbeddingRuntimeConfig) -> list[dict[str, str]]:
    return [
        {
            "Setting": "QDRANT_URL",
            "Status": "Configured" if bool(runtime_config.qdrant_url) else "Missing",
            "Used for": "Vector storage and retrieval",
        },
        {
            "Setting": "GITHUB_TOKEN or GITHUB_WIKI_TOKEN",
            "Status": "Configured" if bool(os.getenv("GITHUB_WIKI_TOKEN") or os.getenv("GITHUB_TOKEN")) else "Missing",
            "Used for": "Cloning and pushing the repo wiki",
        },
        {
            "Setting": "GITHUB_WEBHOOK_SECRET",
            "Status": "Configured" if bool(os.getenv("GITHUB_WEBHOOK_SECRET")) else "Missing",
            "Used for": "GitHub webhook signature verification",
        },
        {
            "Setting": "WEBHOOK_PUBLIC_BASE_URL",
            "Status": "Configured" if bool(os.getenv("WEBHOOK_PUBLIC_BASE_URL")) else "Optional",
            "Used for": "Showing the exact public webhook URL in this UI",
        },
    ]


st.set_page_config(page_title="Code To Doc - RAG Documentation", page_icon=":book:", layout="wide")
st.title("Codebase Parser")
st.caption(
    "Index GitHub repositories into Qdrant with CST embeddings, then retrieve that context to generate markdown documentation with a local Hugging Face model."
)

runtime_config = EmbeddingRuntimeConfig.from_env()
generate_tab, webhook_tab = st.tabs(["Generate Docs", "Webhook Setup"])

with generate_tab:
    with st.form("repo_form"):
        repository_url = st.text_input(
            "Repository URL",
            placeholder="https://github.com/owner/repo",
            help="You can also enter owner/repo",
        )
        ref = st.text_input("Branch / Tag / Commit (optional)", placeholder="main")
        github_token = st.text_input(
            "GitHub Token (optional)",
            type="password",
            value=os.getenv("GITHUB_TOKEN", ""),
            help="Needed for private repos or to avoid rate limits.",
        )
        publish_to_wiki = st.checkbox(
            "Publish generated documentation to the repository wiki",
            value=False,
            help="Uses GITHUB_WIKI_TOKEN or GITHUB_TOKEN to push a new wiki page after generation.",
        )
        wiki_page_title = st.text_input(
            "Wiki page title (optional)",
            placeholder="Manual Docs owner/repo main 20260331-120000",
            help="Leave blank to auto-generate a unique page title.",
        )

        submitted = st.form_submit_button("Run")

    if submitted:
        if not repository_url.strip():
            st.error("Repository URL is required.")
        else:
            try:
                workflow = build_documentation_workflow(
                    runtime_config=runtime_config,
                    github_token=github_token.strip() or None,
                )

                with st.spinner("Checking Qdrant, retrieving context, and generating documentation locally..."):
                    result = workflow.run(repository_url=repository_url.strip(), ref=ref.strip() or None)

                if result.indexed_now:
                    st.success("Repository indexed and documentation generated successfully.")
                else:
                    st.success("Repository found in Qdrant. Documentation generated from stored context.")

                if result.index_result is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Fetched Files", result.index_result.fetched_files)
                    col2.metric("Parsed Files", len(result.index_result.parsed_files))
                    col3.metric("Skipped Files", len(result.index_result.skipped_files))
                    col4.metric("Stored Embeddings", len(result.index_result.stored_ids))
                    st.caption(f"Qdrant collection: `{result.index_result.collection_name}`")
                else:
                    st.info("Repository was already indexed, so fetch/parse/embed steps were skipped.")
                    st.caption(f"Qdrant collection: `{runtime_config.qdrant_collection_name}`")

                st.caption(f"Documentation model: `{runtime_config.docs_model_name}`")

                if publish_to_wiki:
                    publisher = GitHubWikiPublisher(token=github_token.strip() or None)
                    page_title = wiki_page_title.strip() or _default_wiki_page_title(
                        result.documentation.repository,
                        result.documentation.ref,
                    )
                    published_page = publisher.publish_page(
                        repository=result.documentation.repository,
                        page_title=page_title,
                        markdown=build_wiki_page_markdown(
                            repository_url=f"https://github.com/{result.documentation.repository}",
                            ref=result.documentation.ref,
                            markdown=result.documentation.markdown,
                            generated_by="Code To Doc Streamlit",
                        ),
                        commit_message=f"Add generated documentation for {result.documentation.ref or 'HEAD'} (manual run)",
                    )
                    st.success(f"Published wiki page: {published_page.page_title}")
                    st.markdown(f"[Open published wiki page]({published_page.page_url})")

                st.subheader("Generated Documentation")
                st.markdown(result.documentation.markdown)
                st.download_button(
                    "Download Markdown",
                    data=result.documentation.markdown,
                    file_name=f"{result.documentation.repository.replace('/', '-')}-documentation.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

                st.subheader("Retrieved Context")
                st.dataframe(
                    [
                        {
                            "path": document.path,
                            "language": document.language,
                            "score": round(document.score, 4),
                            "root_type": document.metadata.get("root_type"),
                            "node_count": document.metadata.get("node_count"),
                            "has_error": document.metadata.get("has_error"),
                        }
                        for document in result.documentation.retrieved_documents
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                if result.index_result is not None:
                    st.subheader("Parsed File Details")
                    if result.index_result.parsed_files:
                        st.dataframe(
                            [
                                {
                                    "path": parsed.path,
                                    "language": parsed.language,
                                    "root_type": parsed.root_type,
                                    "node_count": parsed.node_count,
                                    "has_error": parsed.has_error,
                                    "cst_length": len(parsed.cst),
                                }
                                for parsed in result.index_result.parsed_files
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No supported source files were parsed.")

                    with st.expander("Skipped Files"):
                        if result.index_result.skipped_files:
                            st.write("\n".join(result.index_result.skipped_files))
                        else:
                            st.write("None")

            except Exception as exc:
                st.error(f"Processing failed: {exc}")

with webhook_tab:
    st.subheader("Webhook Readiness")
    st.dataframe(
        _config_status_rows(runtime_config),
        use_container_width=True,
        hide_index=True,
    )

    public_base_url = st.text_input(
        "Public base URL for the FastAPI service",
        value=os.getenv("WEBHOOK_PUBLIC_BASE_URL", ""),
        placeholder="https://your-domain.example.com",
        help="Used to display the full GitHub webhook URL. Leave blank if you only want the route path.",
    )
    webhook_url = _build_webhook_url(public_base_url)
    st.code(webhook_url, language="text")

    st.subheader("GitHub Setup Steps")
    st.markdown(
        "\n".join(
            [
                "1. Start the API with `uvicorn src.webhook_api:app --reload`.",
                "2. Make that FastAPI service reachable from GitHub. If you are running locally, expose it with a tunnel or deploy it.",
                "3. In the target repository, make sure the **Wiki** feature is enabled in GitHub repository settings.",
                f"4. In GitHub, open **Settings -> Webhooks -> Add webhook** and set the payload URL to `{webhook_url}`.",
                "5. Set **Content type** to `application/json`.",
                "6. Paste the same value into the GitHub webhook secret field that you set in `GITHUB_WEBHOOK_SECRET`.",
                "7. Choose **Just the push event** so documentation is regenerated on every push.",
                "8. Save the webhook, then push a commit to the repository and confirm that a new wiki page is created.",
            ]
        )
    )

    st.subheader("Environment Variables")
    st.code(
        "\n".join(
            [
                "GITHUB_TOKEN=...",
                "GITHUB_WIKI_TOKEN=...  # optional if GITHUB_TOKEN already has repo/wiki access",
                "GITHUB_WEBHOOK_SECRET=...",
                "WEBHOOK_PUBLIC_BASE_URL=https://your-domain.example.com",
                "QDRANT_URL=...",
                "QDRANT_API_KEY=...",
            ]
        ),
        language="bash",
    )

    with st.expander("Example GitHub webhook test payload"):
        st.code(
            "\n".join(
                [
                    "curl -X POST \\",
                    f"  {webhook_url} \\",
                    "  -H \"Content-Type: application/json\" \\",
                    "  -H \"X-GitHub-Event: push\" \\",
                    "  -d '{",
                    '    "ref": "refs/heads/main",',
                    '    "after": "0123456789abcdef",',
                    '    "repository": {',
                    '      "full_name": "owner/repo",',
                    '      "html_url": "https://github.com/owner/repo"',
                    "    }",
                    "  }'",
                ]
            ),
            language="bash",
        )
