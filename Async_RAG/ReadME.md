Below is a `README.md` you can drop next to your `ollama-rag-run.py`. It mirrors the style of your existing docs but focuses on the retrieval API, splitting options, metadata mappings, vector-store choices, and persistence.

---

# Ollama RAG Retriever

This package provides two paired functions, `build_retriever` and `build_retriever_async`, to turn a set of PDF/CSV documents into a retrieval-augmented LLM pipeline powered by Ollama embeddings and LangChain FAISS/Chroma vector stores.

Each chunk you retrieve carries rich metadata (e.g. source filename, party label, CSV columns), and you can choose:

* **How to split** your text (recursive vs. semantic sentence splitting)
* **Which embedder** to use (default Ollama or your own)
* **Which vectorstore** to build (FAISS or Chroma)
* **Whether to persist** the index on disk

---

## Installation

```bash
# 1) Install Python deps
pip install pandas numpy tqdm ollama langchain langchain-community chromadb faiss-cpu spacy nest_asyncio

# 2) (Optional) Install and download a SpaCy model for semantic splitting:
python -m spacy download en_core_web_sm

# 3) Ensure your Ollama daemon is running:
ollama serve
```

---

## 1 · `build_retriever()` (synchronous)

```python
from ollama_rag_run import build_retriever

retriever = build_retriever(
    pdf_paths=["afd.pdf","cdu.pdf"],
    pdf_metadata_map={"afd.pdf":{"party":"AfD"}, "cdu.pdf":{"party":"CDU"}},
    csv_paths=["speeches.csv"],
    csv_text_column="text.ori",
    csv_metadata_fields=["speaker","date"],
    splitter_type="semantic",
    spacy_pipeline="en_core_web_sm",
    chunk_size=800,
    chunk_overlap=100,
    embeddings_factory=lambda: MyCustomEmbeddings(),
    vectorstore="chroma",
    vectorstore_kwargs={"persist_directory":"./chroma_db"},
    k=5,
)
```

| Parameter             | Default        | Purpose                                                                                         |
| --------------------- | -------------- | ----------------------------------------------------------------------------------------------- |
| `pdf_paths`,          | —              | List of PDF paths to index.                                                                     |
| `pdf_metadata_map`    | `{}`           | `{"path.pdf": {...}}` extra metadata per PDF                                                    |
| `csv_paths`           | —              | List of CSV paths.                                                                              |
| `csv_text_column`     | `"text"`       | Which column to chunk in each CSV.                                                              |
| `csv_metadata_fields` | `None`         | List of other CSV columns to attach as metadata.                                                |
| `splitter_type`       | `"recursive"`  | `"recursive"` (char chunks) or `"semantic"` (spaCy sentence chunks).                            |
| `chunk_size`,         | `1000`, `200`  | Chunk parameters for the splitter.                                                              |
| `embeddings`          | Ollama default | Pass your own instance, or use `embeddings_factory` for custom backends (e.g. Bedrock, Ollama). |
| `vectorstore`         | `"faiss"`      | `"faiss"` or `"chroma"`                                                                         |
| `vectorstore_kwargs`  | `{}`           | Extra params—e.g. Chroma’s `{"persist_directory": "./db"}`                                      |
| `k`                   | `4`            | Number of hits to return per query.                                                             |

---

## 2 · `build_retriever_async()` (asynchronous)

```python
from ollama_rag_run import build_retriever_async

# inside an async context or Jupyter cell:
retriever = await build_retriever_async(
    pdf_paths=["afd.pdf","cdu.pdf"],
    pdf_metadata_map={"afd.pdf":{"party":"AfD"}, "cdu.pdf":{"party":"CDU"}},
    csv_paths=["speeches.csv"],
    csv_text_column="text.ori",
    csv_metadata_fields=["speaker","date"],
    splitter_type="recursive",
    chunk_size=1200,
    chunk_overlap=300,
    model="nomic-embed-text",
    vectorstore="faiss",
    batch_size=64,
    max_concurrency=8,
    persist_directory="./faiss_store",
)
```

| Parameter             | Default              | Purpose                                                    |
| --------------------- | -------------------- | ---------------------------------------------------------- |
| All the same as above |                      |                                                            |
| `model`               | `"nomic-embed-text"` | Ollama model tag for embeddings.                           |
| `batch_size`          | `16`                 | How many texts per embed‐call (larger = fewer HTTP trips). |
| `max_concurrency`     | `8`                  | Parallel embed requests.                                   |
| `persist_directory`   | `None`               | If set, saves the built FAISS/Chroma store to disk.        |

---

## 3 · Retrieving & Integrating into LLM

Once you have a `retriever`, you can:

```python
# get top-k chunks
docs = retriever.get_relevant_documents("Your query here")
for d in docs:
    print(d.page_content[:200], "...", d.metadata)

# integrate into your LLM prompt:
from ollama_rag_run import run_analysis
result_df = run_analysis(
    df,                            # your DataFrame of speeches
    text_column="text.ori",
    retrievers=retriever,          # pass the store here
    workers=4,
    prompt_template=(
        "Context snippets:\n{context}\n\n"
        "Which party said: {text}? Answer with party only."
    ),
    json_keys=("party",),
)
```

Here each `Document` carries:

* `page_content`: the text chunk
* `metadata`: a dict with keys

  * `"source"` (path)
  * any fields from `pdf_metadata_map` or your `csv_metadata_fields`

---

## 4 · Persistence & Reuse

### FAISS

```python
# Save
store.save_local("./faiss_store")

# Reload later
from langchain.vectorstores import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings
emb = OllamaEmbeddings(model="nomic-embed-text")
store = FAISS.load_local("./faiss_store", emb)
retriever = store.as_retriever(search_kwargs={"k":5})
```

### Chroma

```python
# Already persisted if you passed `persist_directory` + `.persist()`
from langchain.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
emb = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(persist_directory="./chroma_db", embedding_function=emb)
retriever = store.as_retriever(search_kwargs={"k":5})
```

---

## 5 · Splitting Strategies

* **RecursiveCharacterTextSplitter** (default): fixed‐size chunks plus overlap. Fast, simple.
* **SpacyTextSplitter** (semantic): splits on sentence boundaries. Preserves meaning but depends on SpaCy pipeline.

Use `splitter_type="semantic"` if you need sentence coherence; otherwise stick with `"recursive"` for maximum throughput.

---

## License & Contributing

Please see [LICENSE](./LICENSE) and feel free to open issues or pull requests for enhancements!

---

That’s it—happy retriever‐augmented querying!

