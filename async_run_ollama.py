# parallel_llama_df_analysis.py
"""
Asynchronous helpers for Ollama models**
============================================================================

"""

from __future__ import annotations
import random
import argparse
import asyncio
import json
import sys
import textwrap
from typing import Any, Callable, Dict, List, Optional, Union

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
    batch_size: int,
    temperature: float,
    max_tokens: int,
) -> None:
    client = AsyncClient()
    try:
        buf_tasks: list[asyncio.Task] = []
        buf_idx:   list[int] = []

        async def _flush() -> None:
            for ridx, reply in zip(buf_idx, await asyncio.gather(*buf_tasks)):
                if fanout:
                    out[ridx][model_name] = reply
                else:
                    out[ridx] = reply
            buf_tasks.clear()
            buf_idx.clear()

        for idx, row in tqdm(
            chunk.iterrows(),
            total=len(chunk),
            leave=False,
            desc=f"worker-{model_name}",
        ):
            user_text = str(row[text_column])
            prompt = (
                prompt_template.format(text=user_text)
                if prompt_template else
                user_text
            )

            if json_keys:
                messages = [
                    {"role": "system", "content": _json_system_prompt(json_keys)},
                    {"role": "user",   "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

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

            if len(buf_tasks) == batch_size:
                await _flush()

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
    temperature: float = 0.9,
    max_tokens: int = MAX_TOKENS,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"{text_column!r} not found")
    if isinstance(model_names, str):
        model_names = [model_names] * workers
    if len(model_names) != workers:
        raise ValueError("len(model_names) must equal workers")

    sem = aSYNC_SEM(max_concurrent_calls or workers)
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
            result["analysis"] = buf

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
    temperature: float = 0.9,
    max_tokens: int = MAX_TOKENS,
) -> pd.DataFrame:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = analyze_dataframe(
        df=df,
        text_column=text_column,
        workers=workers,
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
    num_predict: int = MAX_TOKENS,
    temperature: float = 0.9,
) -> Dict[str, str | None]:
    keys = ",".join(f'"{k}": string|null' for k in json_fields)
    system = (
        "You are a JSON-only API. Respond with exactly one JSON object "
        f"{{{keys}}}. If unsure write \"Unclear\"."
    )
    user = f'Title: "{title}"\nReturn only the missing keys: {", ".join(missing)}.'
    raw = await _chat_single(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model_name=model_name,
        num_predict=num_predict,
        temperature=temperature,
    )
    return _to_json(raw, json_fields)

async def _flush(tasks, idxs, chunk, col_map, fanout, model_name)->int:
    filled=0
    for ridx, parsed in zip(idxs, await asyncio.gather(*tasks)):
        for k,col in col_map.items():
            target = f"{col}_{model_name}" if fanout else col
            if _keep(parsed.get(k)):
                chunk.at[ridx, target] = parsed[k]; filled+=1
    tasks.clear(); idxs.clear(); return filled

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
    temperature: float,
    max_tokens: int,
) -> int:
    client, filled = AsyncClient(), 0
    try:
        buf_tasks: list[asyncio.Task] = []; buf_idx: list[int] = []
        for idx, row in tqdm(sub.iterrows(), total=len(sub), leave=False):
            known = {k: row.get(k.lower(), pd.NA) for k in json_fields}
            if "City" in json_fields and pd.isna(known.get("City")):
                known["City"] = row.get("location", pd.NA)
            missing = [k for k, v in known.items() if pd.isna(v)]

            for k, v in known.items():
                if _keep(v):
                    target = f"{col_map[k]}_{model_name}" if fanout else col_map[k]
                    sub.at[idx, target] = v
                    filled += 1

            if not missing:
                continue

            async with semaphore:
                buf_tasks.append(
                    asyncio.create_task(
                        _infer_json(
                            client,
                            row[title_col],
                            missing,
                            json_fields,
                            model_name=model_name,
                            num_predict=max_tokens,
                            temperature=temperature,
                        )
                    )
                )
                buf_idx.append(idx)

            if len(buf_tasks) == batch_size:
                filled += await _flush(buf_tasks, buf_idx, sub, col_map, fanout, model_name)

        if buf_tasks:
            filled += await _flush(buf_tasks, buf_idx, sub, col_map, fanout, model_name)
    finally:
        await _maybe_aclose(client)
    return filled


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
    model_names: List[str] | str,
    fanout: bool,
    temperature: float,
    max_tokens: int,
):
    if isinstance(model_names, str):
        model_names = [model_names] * workers
    if len(model_names) != workers:
        raise ValueError("len(model_names) must equal workers")

    first = True
    semaphore = aSYNC_SEM(workers)
    itr = pd.read_csv(input_csv, chunksize=chunk_size)
    for chunk in tqdm(itr, desc="CSV chunks"):
        for key, col in col_map.items():
            cols = [f"{col}_{m}" for m in model_names] if fanout else [col]
            for c in cols:
                if c not in chunk:
                    chunk[c] = pd.NA

        parts = np.array_split(chunk, workers)
        await asyncio.gather(*[
            _process_subchunk(
                p,
                title_col=title_col,
                batch_size=batch_size,
                semaphore=semaphore,
                json_fields=json_fields,
                col_map=col_map,
                model_name=model_names[i],
                fanout=fanout,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for i, p in enumerate(parts)
        ])

        chunk.to_csv(output_csv, mode="a", header=first, index=False)
        first = False


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
    model_names: List[str] | str = MODEL_NAME,
    fanout: bool = False,
    temperature: float = 0.9,
    max_tokens: int = MAX_TOKENS,
) -> None:
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
        model_names=model_names,
        fanout=fanout,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if loop and loop.is_running():
        import nest_asyncio; nest_asyncio.apply()
        return loop.run_until_complete(coro)
    return asyncio.run(coro)
# ---------------------------------------------------------------------------
# Part 3 – Create Fake Survey Responses 
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Create the Fake Survey Takers (generate_fake_survey_df())
# ---------------------------------------------------------------------------
SpecType = Union[
    List[Any],                    # uniform list
    Dict[Any, float],             # weighted dict
    Callable[[], Any],            # custom function
    Dict[str, Any]                # conditional spec: {'depends_on': str, 'distributions': {...}}
]

def generate_fake_survey_df(
    n: int,
    *,
    seed: Optional[int] = None,
    characteristics: Dict[str, SpecType],
    fixed_values: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Generate a DataFrame of fake survey takers with support for conditional distributions.

    Parameters:
    - n: Number of rows (fake respondents).
    - seed: Optional random seed.
    - characteristics: mapping col name -> spec, where spec is:
        • list of values (uniform)
        • dict value->weight (weighted)
        • callable() -> value
        • conditional dict:
            {
              "depends_on": "<other_column>",
              "distributions": {
                <parent_value>: <spec (list/dict/callable)>,
                ...
              }
            }
    - fixed_values: mapping col name -> single fixed value

    Returns:
    - pd.DataFrame with columns: ID, each characteristic, plus any fixed columns.
    """
    if seed is not None:
        random.seed(seed)

    # split unconditional vs conditional
    unconditional: Dict[str, SpecType] = {}
    conditional: Dict[str, Dict[str, Any]] = {}
    for col, spec in characteristics.items():
        if isinstance(spec, dict) and "depends_on" in spec and "distributions" in spec:
            conditional[col] = spec  # type: ignore
        else:
            unconditional[col] = spec

    def _draw_from_spec(spec: SpecType, k: int) -> List[Any]:
        """Helper: draw k samples from a spec."""
        if isinstance(spec, dict) and not ("depends_on" in spec and "distributions" in spec):
            # weighted dict
            choices, weights = zip(*spec.items())  # type: ignore
            return random.choices(choices, weights=weights, k=k)
        elif isinstance(spec, list):
            return random.choices(spec, k=k)
        elif callable(spec):
            return [spec() for _ in range(k)]
        else:
            raise ValueError(f"Invalid spec: {spec}")

    # build data container
    data: Dict[str, List[Any]] = {"ID": list(range(1, n + 1))}

    # generate all unconditional columns
    for col, spec in unconditional.items():
        data[col] = _draw_from_spec(spec, n)

    # now generate conditional columns
    for col, cond in conditional.items():
        parent = cond["depends_on"]
        dists: Dict[Any, SpecType] = cond["distributions"]
        vals: List[Any] = []
        for i in range(n):
            parent_val = data[parent][i]
            if parent_val not in dists:
                raise KeyError(f"No distribution for parent value '{parent_val}' in column '{col}'")
            spec_for_parent = dists[parent_val]
            # draw one sample from that spec
            single = _draw_from_spec(spec_for_parent, 1)[0]
            vals.append(single)
        data[col] = vals

    # add fixed columns
    if fixed_values:
        for col, val in fixed_values.items():
            data[col] = [val] * n

    return pd.DataFrame(data)
# ---------------------------------------------------------------------------
# Initialization of Survey Workers (simulate_survey_responses())
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Survey Worker
# ---------------------------------------------------------------------------

async def _worker_survey(
    worker_id: int,
    chunk: pd.DataFrame,
    *,
    questions: Dict[str, Optional[List[str]]],
    out: List[Dict[str, str]],
    semaphore: asyncio.Semaphore,
    model_name: str,
    temperature: float,
    batch_size: int,
    max_tokens: int,
) -> None:
    client = AsyncClient()
    try:
        buf_tasks: List[asyncio.Task] = []
        buf_meta:  List[tuple[int, str]] = []

        # progress bar for this worker
        bar = tqdm(
            total=len(chunk),
            desc=f"worker-{worker_id}-{model_name}",
            position=worker_id+1,
            leave=True,
            dynamic_ncols=True
        )

        async def _flush():
            replies = await asyncio.gather(*buf_tasks)
            for (ridx, q), reply in zip(buf_meta, replies):
                out[ridx][q] = reply
            buf_tasks.clear()
            buf_meta.clear()

        for idx, row in chunk.iterrows():
            # build system prompt with characteristics
            char_parts = [f"{col}={row[col]}" for col in row.index]
            base_prompt = (
                "You are a fake survey taker with these characteristics: "
                + "; ".join(char_parts)
            )

            for question, options in questions.items():
                # build question prompt
                if options is not None:
                    opts_str = ", ".join(options)
                    user_msg = f"{question}\nOptions: {opts_str}\nRespond to each question by outputting ONLY the chosen option (no extra text)."
                else:
                    user_msg = question

                messages = [
                    {"role": "system", "content": base_prompt},
                    {"role": "user",   "content": user_msg},
                ]

                # schedule chat for both open and choice
                async with semaphore:
                    buf_tasks.append(
                        asyncio.create_task(
                            _chat_single(
                                client,
                                messages,
                                model_name=model_name,
                                num_predict=max_tokens,
                                temperature=temperature,
                            )
                        )
                    )
                    buf_meta.append((idx, question))

                if len(buf_tasks) >= batch_size:
                    await _flush()

            bar.update(1)

        if buf_tasks:
            await _flush()
        bar.close()

    finally:
        await _maybe_aclose(client)

# ---------------------------------------------------------------------------
# Simulation Functions
# ---------------------------------------------------------------------------

async def simulate_survey_responses(
    df: pd.DataFrame,
    questions: Dict[str, Optional[List[str]]],
    workers: int = 3,
    chunk_size: Optional[int] = None,
    batch_size: int = 4,
    max_concurrent_calls: Optional[int] = None,
    max_tokens: int = 256,
    *,
    model_names: Union[List[str], str] = MODEL_NAME,
    temperature: float = 0.9,
) -> pd.DataFrame:
    if isinstance(model_names, str):
        model_names = [model_names] * workers
    if len(model_names) != workers:
        raise ValueError("len(model_names) must equal workers")

    sem = asyncio.Semaphore(max_concurrent_calls or workers)
    buf: List[Dict[str, str]] = [{q: "" for q in questions} for _ in range(len(df))]

    for start in tqdm(
        range(0, len(df), chunk_size or len(df)),
        desc="DF chunks",
        position=0,
        leave=True
    ):
        sub = df.iloc[start : start + (chunk_size or len(df))]
        subs = np.array_split(sub, workers)
        await asyncio.gather(*[
            _worker_survey(
                i,
                subs[i],
                questions=questions,
                out=buf,
                semaphore=sem,
                model_name=model_names[i],
                temperature=temperature,
                batch_size=batch_size,
                max_tokens=max_tokens,
            )
            for i in range(workers)
        ])

    result = df.copy()
    for q in questions:
        result[q] = [rowbuf[q] for rowbuf in buf]
    return result


def run_survey_responses(
    df: pd.DataFrame,
    questions: Dict[str, Optional[List[str]]],
    workers: int = 3,
    chunk_size: Optional[int] = None,
    batch_size: int = 4,
    max_concurrent_calls: Optional[int] = None,
    max_tokens: int = 256,
    *,
    model_names: Union[List[str], str] = MODEL_NAME,
    temperature: float = 0.9,
) -> pd.DataFrame:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = simulate_survey_responses(
        df,
        questions,
        workers=workers,
        chunk_size=chunk_size,
        batch_size=batch_size,
        max_concurrent_calls=max_concurrent_calls,
        max_tokens=max_tokens,
        model_names=model_names,
        temperature=temperature,
    )
    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
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
