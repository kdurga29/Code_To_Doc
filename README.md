# Code To Doc (Developing)

## CST Embeddings + Qdrant Vector Store

The project now supports:
- Tree-sitter CST parsing for supported source files.
- CST serialization per file.
- Hugging Face embedding generation via LangChain (default: `BAAI/bge-base-en-v1.5`).
- LangGraph workflow orchestration + Qdrant upsert for CST embeddings.

## Streamlit

```bash
streamlit run streamlit_app.py
```