from typing import (
    List, Optional, Union, Any, Callable, Dict
)
import json
import asyncio
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os

from ollama import AsyncClient
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS, Chroma

async def _maybe_aclose(client: AsyncClient) -> None:
    try:
        # preferred async shutdown
        await client.aclose()
    except AttributeError:
        # fall back to sync if provided
        try:
            fn = client.close
        except AttributeError:
            return
        if asyncio.iscoroutinefunction(fn):
            await fn()
        else:
            fn()
    except Exception:
        # any other error during close is non-fatal
        return


def build_retriever(
    *,
    pdf_paths: Optional[List[str]] = None,
    csv_paths: Optional[List[str]] = None,
    csv_text_column: str = "text",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 4,
    # —— Embedding selection —— #
    embeddings: Optional[Any] = None,
    embeddings_factory: Optional[Callable[[], Any]] = None,
    default_ollama_model: str = "nomic-embed-text",
    # —— Vector store choice —— #
    vectorstore: str = "faiss",
    vectorstore_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[FAISS, Chroma]:
    # 1) pick your embedder (default is Ollama)
    if embeddings is not None:
        embedder = embeddings
    elif embeddings_factory is not None:
        embedder = embeddings_factory()
    else:
        embedder = OllamaEmbeddings(model=default_ollama_model)

    # 2) load & split documents with progress bars
    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if pdf_paths:
        for pdf in tqdm(pdf_paths, desc="Loading PDFs", unit="file"):
            raw_docs = PyPDFLoader(pdf).load()
            pages = splitter.split_documents(raw_docs)
            for pg in pages:
                pg.metadata["source"] = pdf
            docs.extend(pages)

    if csv_paths:
        for csv in tqdm(csv_paths, desc="Loading CSVs", unit="file"):
            df = pd.read_csv(csv)
            if csv_text_column not in df.columns:
                raise ValueError(f"Column {csv_text_column!r} not found in {csv}")
            for i, txt in enumerate(
                tqdm(df[csv_text_column].astype(str),
                     desc=f"Splitting rows from {os.path.basename(csv)}",
                     unit="row"),
                start=1,
            ):
                for chunk in splitter.split_text(txt):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"source": csv, "row": i}
                    ))

    if not docs:
        raise ValueError("No documents to index — pass pdf_paths and/or csv_paths")

    # 3) embed documents with progress
    texts = [d.page_content for d in docs]
    embeddings_list: List[List[float]] = []
    for txt in tqdm(texts, desc="Embedding documents", unit="chunk"):
        emb = embedder.embed_documents([txt])[0]
        embeddings_list.append(emb)

    # 4) build the chosen vector store
    vs_kwargs = vectorstore_kwargs or {}
    if vectorstore.lower() == "faiss":
        index = FAISS.from_embeddings(embeddings_list, docs, **vs_kwargs)
    else:
        index = Chroma.from_documents(docs, embedder, **vs_kwargs)

    # 5) return a retriever
    return index.as_retriever(search_kwargs={"k": k})


async def build_retriever_async(
    *,
    pdf_paths: Optional[List[str]] = None,
    pdf_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    csv_paths: Optional[List[str]] = None,
    csv_text_column: str = "text",
    csv_metadata_fields: Optional[List[str]] = None,
    splitter_type: str = "recursive",            # or "semantic"
    spacy_pipeline: str = "en_core_web_sm",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    model: str = "nomic-embed-text",
    vectorstore: str = "faiss",
    vectorstore_kwargs: Optional[Dict[str, Any]] = None,
    batch_size: int = 16,
    max_concurrency: int = 8,
    persist_directory: Optional[str] = None,      # optional persistence
) -> Union[FAISS, Chroma]:
    # 1) Choose splitter
    if splitter_type == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif splitter_type == "semantic":
        splitter = SpacyTextSplitter(
            pipeline=spacy_pipeline,
            max_length=chunk_size,
        )
    else:
        raise ValueError("splitter_type must be 'recursive' or 'semantic'")

    # 2) Load & split
    docs: List[Document] = []
    if pdf_paths:
        pdf_metadata_map = pdf_metadata_map or {}
        for p in tqdm(pdf_paths, desc="Loading PDFs", unit="file"):
            pages = splitter.split_documents(PyPDFLoader(p).load())
            for pg in pages:
                pg.metadata["source"] = p
                pg.metadata.update(pdf_metadata_map.get(p, {}))
            docs.extend(pages)

    if csv_paths:
        for p in tqdm(csv_paths, desc="Loading CSVs", unit="file"):
            df = pd.read_csv(p)
            for _, row in tqdm(
                df.iterrows(), total=len(df),
                desc=f"Splitting rows from {os.path.basename(p)}", unit="row"
            ):
                text = str(row[csv_text_column])
                for chunk in splitter.split_text(text):
                    md: Dict[str, Any] = {"source": p}
                    for fld in (csv_metadata_fields or []):
                        if fld in row:
                            md[fld] = row[fld]
                    docs.append(Document(page_content=chunk, metadata=md))

    if not docs:
        raise ValueError("No documents provided")

    # 3) Embed in parallel batches
    texts = [d.page_content for d in docs]
    client = AsyncClient()
    sem = asyncio.Semaphore(max_concurrency)

    async def embed_batch(batch: List[str]):
        async with sem:
            resp = await client.embed(model=model, input=batch)
            return resp.get("embeddings", resp.embeddings)

    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    tasks = [asyncio.create_task(embed_batch(b)) for b in batches]

    embeddings: List[List[float]] = []
    for fut in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Embedding batches",
        unit="batch"
    ):
        embeddings.extend(await fut)

    await _maybe_aclose(client)

    # 4) Build & optionally persist
    vs_kwargs = vectorstore_kwargs or {}
    embedder = OllamaEmbeddings(model=model)

    if vectorstore.lower() == "faiss":
        store = FAISS.from_documents(docs, embedder, **vs_kwargs)
        if persist_directory:
            store.save_local(persist_directory)
    else:
        kwargs = {**vs_kwargs}
        if persist_directory:
            kwargs["persist_directory"] = persist_directory
            store = Chroma.from_documents(docs, embedder, **kwargs)
            store.persist()
        else:
            store = Chroma.from_documents(docs, embedder, **kwargs)

    return store.as_retriever(search_kwargs={"k": len(docs)})
aSYNC_SEM = asyncio.Semaphore

async def _chat_single(
    client: AsyncClient,
    messages: List[Dict[str, str]],
    *,
    model_name: str,
    num_predict: int,
    temperature: float,
) -> str:
    resp = await client.chat(
        model=model_name,
        messages=messages,
        options={"num_predict": num_predict, "temperature": temperature},
    )
    if isinstance(resp, dict):
        return resp["message"]["content"]
    return resp.message.content if hasattr(resp, "message") else str(resp)

def _json_system_prompt(keys: tuple[str, ...]) -> str:
    keys_fmt = ",".join(f'"{k}": string' for k in keys)
    return (
        "You are a JSON-only API. Respond with exactly one JSON object "
        f"{{{keys_fmt}}}. Do not include any commentary or markdown fences. If unsure write \"Unclear\"."
    )


async def _worker_plain(
    chunk: pd.DataFrame,
    *,
    retriever: Optional[Any],
    text_column: str,
    out: List[Union[dict[str, str], str, None]],
    metadata_fields: Optional[List[str]] = None,
    contexts: List[List[str]],
    semaphore: asyncio.Semaphore,
    model_name: str,
    prompt_template: Optional[str],
    json_keys: Optional[tuple[str, ...]],
    fanout: bool,
    batch_size: int,
    temperature: float,
    max_tokens: int,
) -> None:
    client = AsyncClient()
    try:
        buf_tasks: List[asyncio.Task] = []
        buf_idx:   List[int]         = []

        async def _flush() -> None:
            results = await asyncio.gather(*buf_tasks)
            for ridx, reply in zip(buf_idx, results):
                if fanout:
                    out[ridx][model_name] = reply
                else:
                    out[ridx] = reply
            buf_tasks.clear()
            buf_idx.clear()

        for idx, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"worker-{model_name}", leave=False):
            user_text = str(row[text_column])
            prompt = prompt_template.format(text=user_text) if prompt_template else user_text

            # build messages, injecting RAG context if present
            messages: List[Dict[str, str]] = []
            if retriever:
                docs = retriever.get_relevant_documents(user_text)

                # decide which metadata keys to include
                keys = metadata_fields or []
                if not keys and docs:
                    # if no explicit list, include all metadata keys
                    keys = list(docs[0].metadata.keys())

                tagged = []
                for d in docs:
                    parts = []
                    for k in keys:
                        v = d.metadata.get(k)
                        parts.append(f"{k}={v}")
                    prefix = "[" + ",".join(parts) + "] "
                    tagged.append(prefix + d.page_content)

                contexts[idx] = tagged
                context_str = "\n\n---\n\n".join(tagged)
                messages.append({
                    "role": "system",
                    "content": f"Context (tagged):\n{context_str}"
                })

            if json_keys:
                messages.append({"role": "system", "content": _json_system_prompt(json_keys)})
                messages.append({"role": "user",   "content": prompt})
            else:
                messages.append({"role": "user",   "content": prompt})

            async with semaphore:
                task = asyncio.create_task(
                    _chat_single(
                        client,
                        messages,
                        model_name=model_name,
                        num_predict=max_tokens,
                        temperature=temperature,
                    )
                )
            buf_tasks.append(task)
            buf_idx.append(idx)

            if len(buf_tasks) >= batch_size:
                await _flush()

        if buf_tasks:
            await _flush()

    finally:
        await _maybe_aclose(client)


async def analyze_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    workers: int = 3,
    retrievers: Optional[Union[Any, List[Any]]] = None,
    context_metadata_fields: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    batch_size: int = 4,
    max_concurrent_calls: Optional[int] = None,
    *,
    model_names: Union[List[str], str] = "llama3.2",
    prompt_template: Optional[str] = None,
    json_keys: Optional[tuple[str, ...]] = None,
    fanout: bool = False,
    temperature: float = 0.9,
    max_tokens: int = 128,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"{text_column!r} not found in DataFrame")

    if isinstance(model_names, str):
        model_names = [model_names] * workers
    if len(model_names) != workers:
        raise ValueError("len(model_names) must equal workers")

    if retrievers is None:
        retriever_list = [None] * workers
    elif isinstance(retrievers, list):
        if len(retrievers) != workers:
            raise ValueError("When passing a list, its length must equal `workers`")
        retriever_list = retrievers
    else:
        retriever_list = [retrievers] * workers

    sem = asyncio.Semaphore(max_concurrent_calls or workers)
    buf: List[Union[dict[str, str], str, None]] = [ {} if fanout else None for _ in range(len(df)) ]
    contexts: List[List[str]] = [[] for _ in range(len(df))]

    for start in tqdm(range(0, len(df), chunk_size or len(df)), desc="DF chunks"):
        sub_df = df.iloc[start : start + (chunk_size or len(df))]
        sub_chunks = np.array_split(sub_df, workers)

        await asyncio.gather(*[
            _worker_plain(
                sub_chunk,
                retriever=retriever_list[i],
                text_column=text_column,
                out=buf,
                metadata_fields=context_metadata_fields,
                contexts=contexts,
                semaphore=sem,
                model_name=model_names[i],
                prompt_template=prompt_template,
                json_keys=json_keys,
                fanout=fanout,
                batch_size=batch_size,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for i, sub_chunk in enumerate(sub_chunks)
        ])

    result = df.copy()
    if fanout:
        for mdl in set(model_names):
            if json_keys:
                for k in json_keys:
                    result[f"{k}_{mdl}"] = None
            else:
                result[f"analysis_{mdl}"] = None

        for i, per_row in enumerate(buf):
            assert isinstance(per_row, dict)
            for mdl, raw in per_row.items():
                if json_keys:
                    parsed = json.loads(raw or "{}")
                    for k in json_keys:
                        result.at[i, f"{k}_{mdl}"] = parsed.get(k)
                else:
                    result.at[i, f"analysis_{mdl}"] = raw
    else:
        if json_keys:
            for k in json_keys:
                result[k] = None
            for i, raw in enumerate(buf):
                parsed = json.loads(raw or "{}")
                for k in json_keys:
                    result.at[i, k] = parsed.get(k)
        else:
            result["analysis"] = buf

    result["context_snippets"] = contexts
    return result


def run_analysis(
    df: pd.DataFrame,
    text_column: str = "text",
    workers: int = 3,
    retrievers: Optional[Union[Any, List[Any]]] = None,
    max_concurrent_calls: Optional[int] = None,
    *,
    chunk_size: Optional[int] = None,
    batch_size: int = 4,
    model_names: Union[List[str], str] = "llama3.2",
    prompt_template: Optional[str] = None,
    json_keys: Optional[tuple[str, ...]] = None,
    fanout: bool = False,
    temperature: float = 0.9,
    max_tokens: int = 128,
    context_metadata_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = analyze_dataframe(
        df=df,
        text_column=text_column,
        workers=workers,
        retrievers=retrievers,
        context_metadata_fields=context_metadata_fields,
        chunk_size=chunk_size,
        batch_size=batch_size,
        max_concurrent_calls=max_concurrent_calls,
        model_names=model_names,
        prompt_template=prompt_template,
        json_keys=json_keys,
        fanout=fanout,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if loop and loop.is_running():
        import nest_asyncio; nest_asyncio.apply()
        return loop.run_until_complete(coro)
    return asyncio.run(coro)
