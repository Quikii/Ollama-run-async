# Ollama async RAG Retriever

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


## 5 · Splitting & Retrieval Deep Dive

When you’re working with long PDFs or CSV text, you need to break (“split”) each document into bite-sized pieces that an embedding model can handle. Later, at query time, you’ll embed your question and find the nearest chunks in that vector space—this is the core of Retrieval-Augmented Generation (RAG). Below is what happens under the hood, and the knobs you can turn.

---

### 5.1 Why Split at All?

- **Token limits**: Ollama (and most embedding models) can only process a few thousand tokens at once.  
- **Granular relevance**: Smaller chunks let the retriever zero in on the exact passage that matches your query.  
- **Memory & speed**: Embedding many medium-sized chunks in parallel is faster than embedding one gigantic document over and over.

---

### 5.2 RecursiveCharacterTextSplitter (default)

- **How it works**  
  1. Treats the text as a long character string.  
  2. Cuts into windows of up to `chunk_size` characters (approximately tokens).  
  3. Rolls forward by `chunk_size – chunk_overlap` each time, so each chunk overlaps the next by `chunk_overlap` characters.  

- **Pros**  
  - Super fast, language-agnostic (no NLP pipeline required).  
  - Overlap preserves context at chunk boundaries.

- **Cons**  
  - May split sentences or even words in half.  
  - Chunks can feel arbitrary, which can dilute embedding quality at true semantic boundaries.

---

### 5.3 SpacyTextSplitter (semantic)

- **How it works**  
  1. Uses SpaCy’s sentence segmentation to break your text into logical sentences.  
  2. Greedily groups consecutive sentences until you reach `chunk_size` tokens.  
  3. Optional small overlap (token-level) can be configured.  

- **Pros**  
  - Chunks respect full sentence boundaries → higher coherence.  
  - Better semantic alignment when you retrieve: each chunk usually contains a complete thought or paragraph.

- **Cons**  
  - Requires loading a SpaCy model (e.g. `en_core_web_sm`) → extra startup cost.  
  - Slower than pure character slicing, especially on large corpora.

---

### 5.4 Tuning `chunk_size` & `chunk_overlap`

| Parameter         | Role                                                          |
| ----------------- | ------------------------------------------------------------- |
| **chunk_size**    | Approximate tokens per chunk. Larger → fewer chunks overall.  |
| **chunk_overlap** | Tokens repeated between adjacent chunks for context glazing.  |

- **Small chunks** (`chunk_size≈500`):  
  + Faster retrieval queries  
  – May lose longer-range context  
- **Large chunks** (`chunk_size≈2000`):  
  + Richer context inside each vector  
  – Fewer chunks → coarser retrieval granularity  
- **Overlap** (`200–400` tokens is typical):  
  Ensures that important sentences near a cut aren’t “orphaned.”

---

### 5.5 From Chunks to Answers

1. **Embedding**  
   Each chunk is passed through your chosen embedder (default Ollama).  
2. **Indexing**  
   Chunks + embeddings go into FAISS or Chroma.  
3. **Querying**  
   - Your input text is embedded  
   - The vector store finds the top-k closest chunk vectors  
4. **Context Injection**  
   Retrieved chunks (and their `metadata`) are stitched together into a system prompt:  

   Context:
   <chunk #1 text>
   ---
   <chunk #2 text>
   …

   User: <your question>


5. **LLM Generation**
   The LLM uses that grounded context—so answers stay factual and traceable.

---

### 5.6 How to Choose?

| Goal                                         | Splitter Type | chunk\_size / overlap | k    |
| -------------------------------------------- | ------------- | --------------------- | ---- |
| **Maximum throughput** (speed & scale)       | recursive     | 1000 / 200            | 4–6  |
| **Sentence-coherent retrieval**              | semantic      | 1200 / 200            | 6–10 |
| **Very fine-grained lookup** (small corpora) | recursive     | 500 / 100             | 8–12 |
| **Rich context per answer**                  | semantic      | 2000 / 400            | 3–5  |

> **Rule of thumb**:
>
> * If your domain text is highly structured (legal, manifestos, academic) → **semantic** splitter.
> * If you need to index tens of thousands of short documents → **recursive** splitter for raw speed.

---

By understanding these choices, you can balance **accuracy**, **throughput**, and **coherence** to tailor your retrieval pipeline exactly to your social-science use case.


---



