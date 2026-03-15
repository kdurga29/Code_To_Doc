"""Streamlit UI for repository parsing + CST embeddings."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from src.embeddings import EmbeddingRuntimeConfig, LangChainEmbeddingModel, QdrantVectorStore, RepositoryCSTEmbeddingIndexer
from src.parser import GitHubRepositoryExtractor, TreeSitterCodeParser

load_dotenv()

st.set_page_config(page_title="Code To Doc - CST Embeddings", page_icon=":book:", layout="wide")
st.title("Codebase Parser")
st.caption("Extract GitHub code, parse CST with Tree-sitter, embed with Hugging Face models via LangChain, and store in Qdrant")

runtime_config = EmbeddingRuntimeConfig.from_env()

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

    submitted = st.form_submit_button("Run")

if submitted:
    if not repository_url.strip():
        st.error("Repository URL is required.")
    else:
        try:
            extractor = GitHubRepositoryExtractor(token=github_token.strip() or None)
            parser = TreeSitterCodeParser()
            indexer = RepositoryCSTEmbeddingIndexer(
                extractor=extractor,
                code_parser=parser,
                embedder=LangChainEmbeddingModel(
                    model_name=runtime_config.model_name,
                    token=runtime_config.hf_token,
                ),
                vector_store=QdrantVectorStore(
                    collection_name=runtime_config.qdrant_collection_name,
                    url=runtime_config.qdrant_url,
                    qdrant_api_key=runtime_config.qdrant_api_key,
                ),
            )

            with st.spinner("Processing repository..."):
                result = indexer.index_repository(repository_url=repository_url.strip(), ref=ref.strip() or None)

            st.success("Repository processed successfully.")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Fetched Files", result.fetched_files)
            col2.metric("Parsed Files", len(result.parsed_files))
            col3.metric("Skipped Files", len(result.skipped_files))
            col4.metric("Stored Embeddings", len(result.stored_ids))
            st.caption(f"Qdrant collection: `{result.collection_name}`")

            st.subheader("Parsed File Details")
            if result.parsed_files:
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
                        for parsed in result.parsed_files
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No supported source files were parsed.")

            with st.expander("Skipped Files"):
                if result.skipped_files:
                    st.write("\n".join(result.skipped_files))
                else:
                    st.write("None")

        except Exception as exc:
            st.error(f"Processing failed: {exc}")
