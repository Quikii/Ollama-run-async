# parallel_llama_df_analysis.py
"""
Asynchronous helpers for Ollama llama3.2 — now with **flexible field mapping**
============================================================================

You can still:
1. **Analyse a DataFrame column** in parallel (`run_analysis`).
2. **Fill missing structured fields** in a large CSV (`fill_missing_fields_from_csv`).

New in this revision
--------------------
* `fill_missing_fields_from_csv()` now lets you override the JSON keys and the
  DataFrame columns they map to via the `json_fields` and `col_map` arguments.
* Works in notebooks and scripts; safely closes any Ollama `AsyncClient`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from typing import Dict, List

import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from ollama import AsyncClient

MODEL_NAME = "llama3.2"
MAX_TOKENS = 128

# ---------------------------------------------------------------------------
# Defaults for the CSV‑stream task
# ---------------------------------------------------------------------------

_JSON_FIELDS_DEFAULT: tuple[str, ...] = ("Occasion", "Institution", "City")
_COL_MAP_DEFAULT: dict[str, str] = {
    "Occasion": "computed_occasion",
    "Institution": "computed_institution",
    "City": "computed_location",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _maybe_aclose(client: AsyncClient) -> None:
    if hasattr(client, "aclose"):
        await client.aclose()
    elif hasattr(client, "close"):
        fn = client.close
        await fn() if asyncio.iscoroutinefunction(fn) else fn()

async def _chat_single(
    client: AsyncClient,
    messages: List[Dict[str, str]],
    *,
    model_name: str = MODEL_NAME,
    num_predict: int = MAX_TOKENS,
    temperature: float = 0.9,
) -> str:
    resp = await client.chat(
        model=model_name,
        messages=messages,
        options={"num_predict": num_predict, "temperature": temperature},
    )
    if isinstance(resp, dict):
        return resp["message"]["content"]
    return resp.message.content if hasattr(resp, "message") else str(resp)
# ---------------------------------------------------------------------------
# Part 1 – Generic DataFrame analysis
# ---------------------------------------------------------------------------

aSYNC_SEM = asyncio.Semaphore

def _json_system_prompt(keys: tuple[str, ...]) -> str:
    keys_fmt = ",".join(f'"{k}": string' for k in keys)
    return (
        "You are a JSON-only API. Respond with exactly one JSON object "
        f"{{{keys_fmt}}}. If unsure write \"Unclear\"."
    )

async def _worker_plain(
    chunk: pd.DataFrame,
    *,
    text_column: str,
    out: List[dict[str, str] | str | None],
    semaphore: aSYNC_SEM,
    model_name: str,
    prompt_template: str | None,
    json_keys: tuple[str, ...] | None,
    fanout: bool,
    batch_size: int,                    # NEW PARAM
) -> None:
    """Process one DataFrame slice on a dedicated AsyncClient.

    Queues up to `batch_size` prompts before awaiting the model response,
    mirroring the batching logic used in the CSV helper.
    """
    client = AsyncClient()
    try:
        buf_tasks: list[asyncio.Task] = []
        buf_idx:   list[int]          = []

        async def _flush() -> None:
            """Await queued tasks and write replies into `out`."""
            for ridx, reply in zip(buf_idx, await asyncio.gather(*buf_tasks)):
                if fanout:
                    out[ridx][model_name] = reply          # type: ignore[index]
                else:
                    out[ridx] = reply                      # type: ignore[index]
            buf_tasks.clear()
            buf_idx.clear()

        # iterate over rows in *this* worker's slice
        for idx, row in chunk.iterrows():
            user_text = str(row[text_column])
            prompt = (
                prompt_template.format(text=user_text)
                if prompt_template else
                user_text
            )

            # build messages
            if json_keys:
                messages = [
                    {"role": "system", "content": _json_system_prompt(json_keys)},
                    {"role": "user",   "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            # create the task under semaphore protection
            async with semaphore:
                task = asyncio.create_task(
                    _chat_single(client, messages, model_name=model_name)
                )
            buf_tasks.append(task)
            buf_idx.append(idx)

            # flush when batch is full
            if len(buf_tasks) == batch_size:
                await _flush()

        # flush remaining prompts
        if buf_tasks:
            await _flush()

    finally:
        await _maybe_aclose(client)

async def analyze_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    workers: int = 3,
    chunk_size: int | None = None,
    batch_size: int = 4,
    max_concurrent_calls: int | None = None,
    *,
    model_names: List[str] | str = MODEL_NAME,
    prompt_template: str | None = None,
    json_keys: tuple[str, ...] | None = None,
    fanout: bool = False,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"{text_column!r} not found")

    if isinstance(model_names, str):
        model_names = [model_names] * workers
    if len(model_names) != workers:
        raise ValueError("len(model_names) must equal workers")

    sem = aSYNC_SEM(max_concurrent_calls or workers)
    
    # buffer stays the same, but declare it *before* the loop
    buf: List[dict[str,str] | str | None] = [ {} if fanout else None for _ in range(len(df)) ]
    
    outer_steps = range(0, len(df), chunk_size or len(df))
    for start in tqdm(outer_steps, desc="DF chunks"):
        sub_df = df.iloc[start : start + (chunk_size or len(df))]
        sub_chunks = np.array_split(sub_df, workers)
    
        await asyncio.gather(*[
            _worker_plain(
                sub_chunk,
                text_column=text_column,
                out=buf,
                semaphore=sem,
                model_name=model_names[i],
                prompt_template=prompt_template,
                json_keys=json_keys,
                fanout=fanout,
                batch_size=batch_size,
            )
            for i, sub_chunk in enumerate(sub_chunks)
        ])

    result = df.copy()

    if fanout:
        # create columns per model (+ per key)
        for mdl in set(model_names):
            if json_keys:
                for k in json_keys:
                    result[f"{k}_{mdl}"] = None
            else:
                result[f"analysis_{mdl}"] = None

        for i, per_row in enumerate(buf):           # type: ignore[assignment]
            assert isinstance(per_row, dict)
            for mdl, raw in per_row.items():
                if json_keys:
                    parsed = _to_json(raw or "", json_keys)
                    for k in json_keys:
                        result.at[i, f"{k}_{mdl}"] = parsed.get(k)
                else:
                    result.at[i, f"analysis_{mdl}"] = raw
    else:
        if json_keys:
            for k in json_keys:
                result[k] = None
            for i, raw in enumerate(buf):
                parsed = _to_json(raw or "", json_keys)
                for k in json_keys:
                    result.at[i, k] = parsed.get(k)
        else:
            result["analysis"] = buf                # type: ignore[arg-type]

    return result

def run_analysis(
    df: pd.DataFrame,
    text_column: str = "text",
    workers: int = 3,
    max_concurrent_calls: int | None = None,
    *,
    chunk_size: int | None = None,
    batch_size: int = 4,
    model_names: List[str] | str = MODEL_NAME,
    prompt_template: str | None = None,
    json_keys: tuple[str, ...] | None = None,
    fanout: bool = False,
) -> pd.DataFrame:
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    coro = analyze_dataframe(
        df, text_column, workers, max_concurrent_calls,
        chunk_size=chunk_size,
        model_names=model_names,
        prompt_template=prompt_template,
        json_keys=json_keys,
        fanout=fanout,
        batch_size=batch_size,
    )

    if loop and loop.is_running():
        import nest_asyncio; nest_asyncio.apply()
        return loop.run_until_complete(coro)
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Part 2 – CSV streaming JSON‑completion task
# ---------------------------------------------------------------------------

def _to_json(raw: str, json_fields: tuple[str, ...]) -> Dict[str, str | None]:
    if "{" in raw and "}" in raw:
        raw = "{" + raw.split("{",1)[1].split("}",1)[0] + "}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {k: None for k in json_fields}

def _keep(v): return pd.notna(v) and v not in ("Unclear", None)

async def _infer_json(
    client: AsyncClient,
    title: str,
    missing: List[str],
    json_fields: tuple[str, ...],
    *,
    model_name: str,
) -> Dict:
    keys = ",".join(f'"{k}": string|null' for k in json_fields)
    system = (
        "You are a JSON-only API. Respond with exactly one JSON object "
        f"{{{keys}}}. If unsure write \"Unclear\"."
    )
    user = f'Title: "{title}"\nReturn only the missing keys: {", ".join(missing)}.'
    raw  = await _chat_single(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model_name=model_name,
    )
    return _to_json(raw, json_fields)

async def _process_subchunk(
    sub: pd.DataFrame,
    *,
    title_col: str,
    batch_size: int,
    semaphore: aSYNC_SEM,
    json_fields: tuple[str, ...],
    col_map: dict[str, str],
    model_name: str,
    fanout: bool,
) -> int:
    client, filled = AsyncClient(), 0
    try:
        buf_tasks: list[asyncio.Task] = []; buf_idx: list[int] = []
        for idx, row in tqdm(sub.iterrows(), total=len(sub), leave=False):
            known = {k: row.get(k.lower(), pd.NA) for k in json_fields}
            if "City" in json_fields and pd.isna(known.get("City")):
                known["City"] = row.get("location", pd.NA)
            missing = [k for k,v in known.items() if pd.isna(v)]

            for k,v in known.items():
                if _keep(v):
                    sub.at[idx, col_map[k]] = v if not fanout else f"{v}"
                    filled += 1

            if not missing: continue

            async with semaphore:
                t = asyncio.create_task(
                        _infer_json(client, row[title_col], missing, json_fields,
                                    model_name=model_name))
            buf_tasks.append(t); buf_idx.append(idx)

            if len(buf_tasks)==batch_size:
                filled += await _flush(buf_tasks, buf_idx, sub, col_map, fanout, model_name)

        if buf_tasks:
            filled += await _flush(buf_tasks, buf_idx, sub, col_map, fanout, model_name)
    finally:
        await _maybe_aclose(client)
    return filled

async def _flush(tasks, idxs, chunk, col_map, fanout, model_name)->int:
    filled=0
    for ridx, parsed in zip(idxs, await asyncio.gather(*tasks)):
        for k,col in col_map.items():
            target = f"{col}_{model_name}" if fanout else col
            if _keep(parsed.get(k)):
                chunk.at[ridx, target] = parsed[k]; filled+=1
    tasks.clear(); idxs.clear(); return filled

async def _process_csv_async(
    input_csv:str, output_csv:str, *, chunk_size:int, workers:int,
    batch_size:int, title_col:str, json_fields:tuple[str,...],
    col_map:dict[str,str], model_names:List[str]|str, fanout:bool,
):
    if isinstance(model_names,str):
        model_names=[model_names]*workers
    if len(model_names)!=workers:
        raise ValueError("len(model_names) must equal workers")

    first=True; semaphore=aSYNC_SEM(workers)
    itr=pd.read_csv(input_csv,chunksize=chunk_size)
    for chunk_no,chunk in enumerate(tqdm(itr,desc="CSV chunks"),start=1):
        # ensure all computed columns exist
        for key,col in col_map.items():
            cols=[f"{col}_{m}" for m in model_names] if fanout else [col]
            for c in cols:
                if c not in chunk: chunk[c]=pd.NA

        parts=np.array_split(chunk,workers)
        counts=await asyncio.gather(*[
            _process_subchunk(
                p,title_col=title_col,batch_size=batch_size,semaphore=semaphore,
                json_fields=json_fields,col_map=col_map,
                model_name=model_names[i],fanout=fanout
            ) for i,p in enumerate(parts)
        ])
        chunk.to_csv(output_csv,mode="a",header=first,index=False); first=False
        print(f"✔ chunk {chunk_no} saved — rows: {len(chunk):,}, filled: {sum(counts):,}")
    print("✅ All chunks processed. Output →",output_csv)

def fill_missing_fields_from_csv(
    *, input_csv:str, output_csv:str="out.csv", chunk_size:int=20_000,
    workers:int=3, batch_size:int=4, title_col:str="title_en",
    json_fields:tuple[str,...]=_JSON_FIELDS_DEFAULT,
    col_map:dict[str,str]=_COL_MAP_DEFAULT,
    model_names:List[str]|str=MODEL_NAME, fanout:bool=False,
):
    loop=None
    try: loop=asyncio.get_running_loop()
    except RuntimeError: pass

    coro=_process_csv_async(
        input_csv,output_csv,chunk_size=chunk_size,workers=workers,
        batch_size=batch_size,title_col=title_col,
        json_fields=json_fields,col_map=col_map,
        model_names=model_names,fanout=fanout,
    )
    if loop and loop.is_running():
        import nest_asyncio; nest_asyncio.apply()
        return loop.run_until_complete(coro)
    return asyncio.run(coro)

# ---------------------------------------------------------------------------
# CLI entry‑point (python parallel_llama_df_analysis.py <input.csv> ...)
# ---------------------------------------------------------------------------

if __name__=="__main__":
    p=argparse.ArgumentParser(prog="parallel_llama_df_analysis",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Analyse DataFrames or enrich CSVs with multiple Ollama models.")
    p.add_argument("input_csv")
    p.add_argument("--output_csv",default="out.csv")
    p.add_argument("--chunk_size",type=int,default=20_000)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--batch_size",type=int,default=4)
    p.add_argument("--title_col",default="title_en")
    p.add_argument("--json_fields",default="Occasion,Institution,City")
    p.add_argument("--col_map",default="")
    p.add_argument("--models",default="llama3.2",
                   help="Comma-sep model tags (one per worker or, with --fanout, all at once)")
    p.add_argument("--fanout",action="store_true",
                   help="If set, every model analyses every row (creates *_<model> columns)")
    args=p.parse_args()

    jf=tuple(k.strip() for k in args.json_fields.split(",") if k.strip())
    cmap={k:f"computed_{k.lower()}" for k in jf}
    if args.col_map:
        for pair in args.col_map.split(","):
            k,v=(s.strip() for s in pair.split(":",1)); cmap[k]=v

    models=[m.strip() for m in args.models.split(",") if m.strip()]
    model_names=models if len(models)>1 or args.fanout else models[0]

    try:
        fill_missing_fields_from_csv(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            chunk_size=args.chunk_size,
            workers=args.workers,
            batch_size=args.batch_size,
            title_col=args.title_col,
            json_fields=jf,
            col_map=cmap,
            model_names=model_names,
            fanout=args.fanout,
        )
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
