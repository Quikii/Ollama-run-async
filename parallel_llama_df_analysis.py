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
from tqdm.auto import tqdm


import numpy as np
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
    """Close `AsyncClient` regardless of installed client version."""
    if hasattr(client, "aclose"):
        await client.aclose()
    elif hasattr(client, "close"):
        fn = client.close
        if asyncio.iscoroutinefunction(fn):
            await fn()
        else:
            fn()


async def _chat_single(
    client: AsyncClient,
    messages: List[Dict[str, str]],
    *,
    num_predict: int = MAX_TOKENS,
    temperature: float = 0.9,
) -> str:
    resp = await client.chat(
        model=MODEL_NAME,
        messages=messages,
        options={"num_predict": num_predict, "temperature": temperature},
    )
    if isinstance(resp, dict):
        return resp["message"]["content"]
    if hasattr(resp, "message") and hasattr(resp.message, "content"):
        return resp.message.content  # newer ollama‑python
    return str(resp)

# ---------------------------------------------------------------------------
# Part 1 – Generic DataFrame analysis
# ---------------------------------------------------------------------------

aSYNC_SEM = asyncio.Semaphore


async def _infer_plain(client: AsyncClient, prompt: str) -> str:
    return await _chat_single(client, [{"role": "user", "content": prompt}])


async def _worker_plain(
    chunk: pd.DataFrame,
    text_column: str,
    out: List[str | None],
    semaphore: aSYNC_SEM,
) -> None:
    client = AsyncClient()
    try:
        for idx, row in chunk.iterrows():
            async with semaphore:
                out[idx] = await _infer_plain(client, row[text_column])
    finally:
        await _maybe_aclose(client)


async def analyze_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    workers: int = 3,
    max_concurrent_calls: int | None = None,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"{text_column!r} not found in DataFrame")

    chunks = np.array_split(df, workers)
    sem = aSYNC_SEM(max_concurrent_calls or workers)
    buf: List[str | None] = [None] * len(df)

    chunk_iter = tqdm(chunks, desc="DF chunks")
    await asyncio.gather(
        *[
            _worker_plain(chunk, text_column, buf, sem)     # type: ignore[arg-type]
            for chunk in chunk_iter
        ]
    )
    df = df.copy()
    df["analysis"] = buf
    return df


def run_analysis(
    df: pd.DataFrame,
    text_column: str = "text",
    workers: int = 3,
    max_concurrent_calls: int | None = None,
) -> pd.DataFrame:
    """Notebook‑friendly wrapper for `analyze_dataframe`."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(
            analyze_dataframe(df, text_column, workers, max_concurrent_calls)
        )
    return asyncio.run(
        analyze_dataframe(df, text_column, workers, max_concurrent_calls)
    )

# ---------------------------------------------------------------------------
# Part 2 – CSV streaming JSON‑completion task
# ---------------------------------------------------------------------------

def _to_json(raw: str, json_fields: tuple[str, ...]) -> Dict[str, str | None]:
    if "{" in raw and "}" in raw:
        raw = "{" + raw.split("{", 1)[1].split("}", 1)[0] + "}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {k: None for k in json_fields}


def _keep(val):  # simple validator
    return pd.notna(val) and val not in ("Unclear", None)


async def _infer_json(
    client: AsyncClient,
    title: str,
    missing: List[str],
    json_fields: tuple[str, ...],
) -> Dict:
    keys = ",".join(f'"{k}": string|null' for k in json_fields)
    system = (
        "You are a JSON‑only API. Respond with exactly one JSON object "
        f"{{{keys}}}. If unsure write \"Unclear\"."
    )
    user = f'Title: "{title}"\nReturn only the missing keys: {", ".join(missing)}.'
    raw = await _chat_single(
        client,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return _to_json(raw, json_fields)


async def _flush(buf_tasks, buf_idx, chunk, col_map) -> int:
    filled = 0
    for row_idx, parsed in zip(buf_idx, await asyncio.gather(*buf_tasks)):
        for key, col in col_map.items():
            if _keep(parsed.get(key)):
                chunk.at[row_idx, col] = parsed[key]
                filled += 1
    buf_tasks.clear()
    buf_idx.clear()
    return filled


async def _process_subchunk(
    sub: pd.DataFrame,
    *,
    title_col: str,
    batch_size: int,
    semaphore: aSYNC_SEM,
    json_fields: tuple[str, ...],
    col_map: dict[str, str],
) -> int:
    client = AsyncClient()
    filled_cells = 0
    try:
        buf_tasks: List[asyncio.Task] = []
        buf_idx: List[int] = []

        for idx, row in tqdm(sub.iterrows(), total=len(sub), leave=False):
            known = {k: row.get(k.lower(), pd.NA) for k in json_fields}
            # fallbacks for City → location etc. kept minimal
            if "City" in json_fields and pd.isna(known.get("City")):
                known["City"] = row.get("location", pd.NA)

            missing = [k for k, v in known.items() if pd.isna(v)]

            for k, v in known.items():
                if _keep(v):
                    sub.at[idx, col_map[k]] = v
                    filled_cells += 1

            if not missing:
                continue

            async with semaphore:
                task = asyncio.create_task(
                    _infer_json(client, row[title_col], missing, json_fields)
                )
            buf_tasks.append(task)
            buf_idx.append(idx)

            if len(buf_tasks) == batch_size:
                filled_cells += await _flush(buf_tasks, buf_idx, sub, col_map)

        if buf_tasks:
            filled_cells += await _flush(buf_tasks, buf_idx, sub, col_map)
    finally:
        await _maybe_aclose(client)
    return filled_cells


async def _process_csv_async(
    input_csv: str,
    output_csv: str,
    *,
    chunk_size: int,
    workers: int,
    batch_size: int,
    title_col: str,
    json_fields: tuple[str, ...],
    col_map: dict[str, str],
):
    first = True
    chunk_no = 0
    semaphore = aSYNC_SEM(workers)

    chunk_iter = pd.read_csv(input_csv, chunksize=chunk_size)
    for chunk_no, chunk in enumerate(tqdm(chunk_iter, desc="CSV chunks"), start=1):
        chunk_no += 1
        for col in col_map.values():
            if col not in chunk:
                chunk[col] = pd.NA

        parts = np.array_split(chunk, workers)
        filled_counts = await asyncio.gather(
            *[
                _process_subchunk(
                    p,
                    title_col=title_col,
                    batch_size=batch_size,
                    semaphore=semaphore,
                    json_fields=json_fields,
                    col_map=col_map,
                )
                for p in parts
            ]
        )
        chunk.to_csv(output_csv, mode="a", header=first, index=False)
        first = False
        print(
            f"✔ chunk {chunk_no} saved — rows: {len(chunk):,}, filled: {sum(filled_counts):,}"
        )

    print("✅ All chunks processed. Output written to", output_csv)


def fill_missing_fields_from_csv(
    *,
    input_csv: str,
    output_csv: str = "out.csv",
    chunk_size: int = 20_000,
    workers: int = 3,
    batch_size: int = 4,
    title_col: str = "title_en",
    json_fields: tuple[str, ...] = _JSON_FIELDS_DEFAULT,
    col_map: dict[str, str] = _COL_MAP_DEFAULT,
):
    """Run the CSV‑stream pipeline (sync) with optional custom field mapping."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = _process_csv_async(
        input_csv,
        output_csv,
        chunk_size=chunk_size,
        workers=workers,
        batch_size=batch_size,
        title_col=title_col,
        json_fields=json_fields,
        col_map=col_map,
    )

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)

    return asyncio.run(coro)

# ---------------------------------------------------------------------------
# CLI entry‑point (python parallel_llama_df_analysis.py <input.csv> ...)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="parallel_llama_df_analysis",
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(
            """
            Two modes in one:
              • Generic DataFrame analysis via Ollama (import and call `run_analysis`).
              • CSV‑streaming fill of structured fields (this script).  

            Examples
            --------
            # default 3‑field mode
            python parallel_llama_df_analysis.py Germany.csv --output_csv out.csv

            # custom two‑field mapping
            python parallel_llama_df_analysis.py Germany.csv \
                --json_fields Occasion,City \
                --col_map Occasion:comp_occasion,City:comp_city
            """
        ),
    )
    parser.add_argument("input_csv", help="Source CSV file")
    parser.add_argument("--output_csv", default="out.csv", help="Destination CSV")
    parser.add_argument("--chunk_size", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--title_col", default="title_en")
    parser.add_argument(
        "--json_fields",
        default="Occasion,Institution,City",
        help="Comma‑separated list of JSON keys to extract",
    )
    parser.add_argument(
        "--col_map",
        default="",
        help="Comma‑sep key:column pairs, e.g. Occasion:comp_occ,City:comp_city",
    )
    args = parser.parse_args()

    # Parse CLI strings → tuples / dict
    json_fields = tuple(k.strip() for k in args.json_fields.split(",") if k.strip())
    if not json_fields:
        sys.exit("json_fields must contain at least one key")

    if args.col_map:
        col_map: dict[str, str] = {}
        for pair in args.col_map.split(","):
            try:
                k, v = (s.strip() for s in pair.split(":", 1))
                col_map[k] = v
            except ValueError:
                sys.exit(f"Bad --col_map entry: {pair!r}")
    else:
        # derive default names if user didn't supply a map
        col_map = {k: f"computed_{k.lower()}" for k in json_fields}

    try:
        fill_missing_fields_from_csv(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            chunk_size=args.chunk_size,
            workers=args.workers,
            batch_size=args.batch_size,
            title_col=args.title_col,
            json_fields=json_fields,
            col_map=col_map,
        )
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
