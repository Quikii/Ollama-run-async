
The code presented makes it simple for researchers in social sciences to run several Large Language Models loaded through ollama at once. This has two reasons: 

1. **Split:** You run several models in parallel on many chunks of documents (the same model several times or different models per chunk). The text documents are stored as rows in a dataframe. This speeds up the computing time.
2. **Fanout:** You run several models in parallel on the same chunks of documents (again, the same model several times or different models per chunk). Again, the text documents are stored as rows in a dataframe. This likewise speeds up the computing time, but primarily allows for convenient comparison of different model outputs.

I present two functions that both can split and fan out over the dataframe, but do so in a slightly different way:
1. **`run_analysis()`:** Allows you to write one prompt, which is then either splits or fans out over the text in the dataframe. The common tasks would be text labeling or sentiment analysis. The answer to the prompt might be conveniently structured in a json object, with specifiable keys.
2. `fill_missing_fields_from_csv()`: Instead of writing a prompt, the second function is specifically designed for information extraction from the text. It also allows for an output in a json format. Crucially, it also handles existing metadata information in the dataframe, so the model only extracts information that is not yet present. 


---

## Installation & model setup

```bash
# 1 · Install the Python package
pip install Ollama-run-async  

# 2 · Have Ollama running and pull the models you plan to use
ollama pull llama3.2            # repeat for other model tags if desired
ollama serve                    # keep this running

# 3 · Python deps
pip install pandas numpy tqdm ollama nest_asyncio
````

If your Ollama server is remote, set  
`export OLLAMA_HOST=http://<ip>:11434` (or `set` on Windows).

---

## Function overview

### 1 · `run_analysis()`

```python
run_analysis(
    df: pd.DataFrame,
    text_column: str = "text",
    workers: int = 3,
    max_concurrent_calls: int | None = None,
    *,                       # keyword-only extras
    model_names: str | list[str] = "llama3.2",
    prompt_template: str | None = None,
    json_keys: tuple[str, ...] | None = None,
    fanout: bool = False,
) -> pd.DataFrame
```

|Parameter|Default|Purpose|
|---|---|---|
|`df`, `text_column`|—|DataFrame and column to process.|
|`workers`|3|Parallel `AsyncClient`s.|
|`model_names`|`"llama3.2"`|Single model for all workers **or** list (one per worker).|
|`prompt_template`|`None`|Format string; `{text}` is replaced by row text.|
|`json_keys`|`None`|If set, model must return one JSON object with these keys; columns are created per key.|
|`fanout`|`False`|`False` → models split the DataFrame. `True` → **every** model analyses **every** row and extra columns are suffixed with `_<model>` (e.g. `label_llama3.2`).|

---

### 2 · `fill_missing_fields_from_csv()`

```python
fill_missing_fields_from_csv(
    input_csv: str,
    output_csv: str = "out.csv",
    chunk_size: int = 20_000,
    workers: int = 3,
    batch_size: int = 4,
    title_col: str = "title_en",
    json_fields: tuple[str, ...] = ("Occasion","Institution","City"),
    col_map: dict[str, str] | None = None,
    model_names: str | list[str] = "llama3.2",
    fanout: bool = False,
) -> None
```

| Parameter     | Default      | Purpose                                                                                      |
| ------------- | ------------ | -------------------------------------------------------------------------------------------- |
| `chunk_size`  | 20 000       | Rows per streamed chunk.                                                                     |
| `workers`     | 3            | Async workers per chunk.                                                                     |
| `batch_size`  | 4            | Prompts queued _per_ worker before awaiting.                                                 |
| `json_fields` | tuple        | Keys expected in JSON answer.                                                                |
| `col_map`     | auto         | Key → column map; omit for `computed_<key>`.                                                 |
| `model_names` | `"llama3.2"` | Tag for all workers **or** list (one per worker).                                            |
| `fanout`      | `False`      | If `True`, every model fills every row and computed columns become `computed_<key>_<model>`. |

---

## Quick examples

```python
from ollama_run_async import run_analysis, fill_missing_fields_from_csv
import pandas as pd

# 1 · Fan-out sentiment scoring with three models
df = pd.read_csv("speeches.csv")
df = run_analysis(
    df,
    text_column="speech",
    workers=3,
    model_names=["llama3.2", "mistral", "phi3.5"],
    prompt_template="Return JSON {\"label\":string,\"prob\":float} for {text}",
    json_keys=("label", "prob"),
    fanout=True,
)
# adds label_llama3.2, prob_llama3.2, label_mistral, …

# 2 · Classic workload split (no fan-out)
scores = run_analysis(df, prompt_template="Summarise: {text}")

# 3 · CSV enrichment with fan-out
fill_missing_fields_from_csv(
    input_csv="events.csv",
    output_csv="events_filled.csv",
    json_fields=("Occasion", "City"),
    col_map={"Occasion":"occ", "City":"place"},
    workers=2,
    model_names=["llama3.2","mistral"],
    fanout=True,
)
```

### CLI

```bash
# every model on every row (fan-out)
python parallel_llama_df_analysis.py events.csv \
       --output_csv out.csv \
       --models llama3.2,mistral \
       --fanout \
       --json_fields Occasion,City
```

---

## How parallelisation works in the code

### One worker ≈ one Ollama “session”

Internally each worker creates its own `AsyncClient`, which translates to an
independent streaming connection and an independent copy of the model held in
GPU (or CPU) memory:

```

worker-0 ─┐ ┌─▶ llama3.2-1B (GPU slot 0)  
worker-1 ─┤ Async → ─┤─▶ llama3.2-1B (GPU slot 0) ← might be shared if the  
worker-2 ─┘ └─▶ llama3.2-1B (GPU slot 0)             model fits multiple

````

* **VRAM footprint per model copy** (≈ numbers for Llama 3.2 1B Instruct variant):
  * **Loaded**: ~2 GB
  * **During generation** (activations): +0.5 GB
* Ollama automatically *re-uses* a loaded model across sessions **as long as it
  fits**, so three workers hitting **llama3.2-1B** typically keep **one** 2 GB
  copy in VRAM, not three. Activations, however, are per-worker, so generation
  spikes can add ~0.5 GB × workers.

### Fan-out vs. split

| Mode | What happens | Memory | When to use |
|------|--------------|--------|-------------|
| **Split** (`fanout=False`) | Each worker/model gets a distinct slice of the DataFrame/CSV. | ① One model copy *per different tag*.<br>② Activations scale with `workers`. | Max throughput when you have *many* rows. |
| **Fan-out** (`fanout=True`) | Every model analyses **every** row – workers loop through rows multiple times. | Same as split, but activations stack for each model on the same row (so memory ≈ *models* × 0.5 GB during generation). | Comparing answers from several models side-by-side. |

### Example

Suppose you run:

```python
run_analysis(
    df,
    workers=4,
    model_names=["llama3.2-1B"]*4,
)
````

- **VRAM baseline**: ~2 GB (one shared copy of the 1 B weights)
    
- **During peak generation**: 2 GB + 4 × 0.5 GB ≈ **4 GB**
    
- **CPU load**: 4 asynchronous decoding threads saturating one GPU.
    

Switch to fan-out with two different models:

```python
run_analysis(
    df,
    workers=2,
    model_names=["llama3.2-1B", "mistral-7B"],
    fanout=True,
)
```

- **Models loaded**: 2 GB (1 B) + 13 GB (7 B) ≈ **15 GB VRAM**
    
- **Peak activations**: +2 × 0.5 GB ≈ **1 GB** extra
    
- Every row produces _two_ sets of outputs: `_llama3.2-1B` and `_mistral-7B`.
    

If VRAM is tight, lower `workers`, set `fanout=False`, or choose smaller  
models. You can also instruct Ollama to keep only one model resident at a time:

```bash
export OLLAMA_MAX_LOADED_MODELS=1
```

Ollama will then swap models in-and-out between requests, trading memory for  
slightly lower throughput.

---

## Troubleshooting

| Issue                                         | Fix                                                                   |
| --------------------------------------------- | --------------------------------------------------------------------- |
| `ResponseError: model not found`              | `ollama pull <model_tag>` on the Ollama host.                         |
| Notebook raises `RuntimeError: asyncio.run()` | Call the high-level helpers – they auto-detect running loops.         |
| VRAM/CPU overload                             | Lower `workers` or `batch_size`, or set `OLLAMA_MAX_LOADED_MODELS=1`. |
