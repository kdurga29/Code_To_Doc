# Code To Doc (Developing)

## CST Embeddings + Qdrant Vector Store

The project now supports:
- Tree-sitter CST parsing for supported source files.
- CST serialization per file.
- Hugging Face embedding generation via LangChain (default: `BAAI/bge-base-en-v1.5`).
- LangGraph workflow orchestration + Qdrant upsert for CST embeddings.
- Repository existence checks against Qdrant before re-indexing.
- RAG-style retrieval over stored CST payloads.
- Markdown documentation generation with a local Hugging Face text-generation model (default: `deepseek-ai/deepseek-coder-1.3b-instruct`).

## Streamlit

```bash
streamlit run streamlit_app.py
```

Set your local documentation model in `.env` before starting the app:

```env
DOCUMENTATION_MODEL_NAME=deepseek-ai/deepseek-coder-1.3b-instruct
DOCUMENTATION_MAX_NEW_TOKENS=512
DOCUMENTATION_TEMPERATURE=0.1
DOCUMENTATION_REPETITION_PENALTY=1.1
DOCUMENTATION_LOAD_IN_4BIT=true
```

The app reuses `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` if the model download needs authentication.
When CUDA is available, it first tries 4-bit loading for lower memory use and falls back to standard loading if quantized loading is unavailable on the machine.

If a repository is already present in the configured Qdrant collection, the app skips re-indexing, retrieves the most relevant CST documents for documentation-focused queries, and sends that context to the configured local model. If the repository is not present yet, the app indexes it first and then runs the same retrieval + generation flow.

The Streamlit UI now also supports:
- Publishing a manually generated documentation run directly to the target repository wiki.
- A dedicated **Webhook Setup** tab with environment readiness checks and GitHub webhook instructions.

## FastAPI GitHub Webhook

Run the webhook API with:

```bash
uvicorn src.webhook_api:app --reload
```

Configure these additional environment variables for webhook-driven wiki publishing:

```env
GITHUB_TOKEN=
GITHUB_WIKI_TOKEN=
GITHUB_WEBHOOK_SECRET=
GITHUB_WIKI_AUTHOR_NAME=Code To Doc Bot
GITHUB_WIKI_AUTHOR_EMAIL=code-to-doc-bot@example.com
```

The webhook endpoint is `POST /webhooks/github`.

- `ping` events return an immediate health-style confirmation.
- `push` events enqueue documentation generation in a background task.
- The generated Markdown is committed as a new page in `<owner>/<repo>.wiki.git`.

The API verifies `X-Hub-Signature-256` when `GITHUB_WEBHOOK_SECRET` is configured.

## Webhook Setup Guide

1. Set the required environment variables:

```env
QDRANT_URL=
QDRANT_API_KEY=
GITHUB_TOKEN=
GITHUB_WIKI_TOKEN=
GITHUB_WEBHOOK_SECRET=
WEBHOOK_PUBLIC_BASE_URL=https://your-domain.example.com
```

2. Start the API:

```bash
uvicorn src.webhook_api:app --host 0.0.0.0 --port 8000
```

3. Make the API publicly reachable from GitHub.
If you are developing locally, use a tunnel or deploy the service.

4. In the target GitHub repository:
- Open `Settings -> General` and make sure **Wikis** are enabled.
- Open `Settings -> Webhooks -> Add webhook`.
- Set the payload URL to `https://your-domain.example.com/webhooks/github`.
- Set content type to `application/json`.
- Enter the same webhook secret you used in `GITHUB_WEBHOOK_SECRET`.
- Select **Just the push event**.

5. Save the webhook and push a commit to the repository.
Each accepted push event will run the documentation pipeline and publish a new wiki page.
