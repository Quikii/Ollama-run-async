<p align="center">
  <img
    src="logo/ollama-async-logo.png"
    alt="Ollama Async Logo"
    style="width:80%; max-width:600px; height:auto;"
  />
</p>


The functions presented in this package make it simple for researchers in social sciences to run several Large Language Models loaded through Ollama over documents stored in a data frame asynchronously (at once). As such, all models used here have to be downloaded through the Ollama interface (https://ollama.com/). With them you are able to:

1. **Analyze** Text Stored in a Dataframe Column
2. **Extract** Missing Metadata Information from a Text Stored in a Dataframe Column
3. **Create** "Fake" LLM Survey Respondents with given characteristics and make them answer your survey questions
4. **Retrieve** Information stored in a separate vector database (Retrieval Augmented Generation (RAG))

The asynchronous running of the functions is based on their ability to:

1. **Split:** You run several models in parallel on many chunks of documents (the same model several times or different models per chunk). The text documents are stored as rows in a dataframe. This speeds up the computing time.
2. **Fanout:** You run several models in parallel on the same chunks of documents (again, the same model several times or different models per chunk). Again, the text documents are stored as rows in a dataframe. This likewise speeds up the computing time, but primarily allows for convenient comparison of different model outputs.

The three basic functions of the package, in Async-RUN, that can either split and/or fan out over the dataframe, but do so in slightly different ways and for slightly different purposes:
1. **`run_analysis()`:** Allows you to write one prompt, which then either splits or fans out over the text in the dataframe. The common tasks would be text labeling or sentiment analysis. The answer to the prompt might be conveniently structured in a JSON object, with specifiable keys.
2. **`fill_missing_fields_from_csv()`**: Instead of writing a prompt, the second function is specifically designed for information extraction from the text (with the primary use case being metadata collection). It also allows for an output in a JSON format. Crucially, it also handles existing metadata information in the dataframe, so the model only extracts information that is not yet present. 
3. **`run_survey_responses()`**: The last function, instead of focusing on analysing existing text, creates fake survey responses based on a set of characteristics. The "fake" respondents are generated and stored in a data frame with the helper function `generate_fake_survey_df()`, which allows the creation of a representative (based on the target population) distribution of characteristics over these respondents. The use case is to have a potentially more accurate distribution of responses to survey questions prior to running the actual survey on real-life respondents. 

In addition to LLM-only processing, this package supports RAG:
1. You index your PDFs/CSVs into a FAISS or Chroma vector store, with each chunk carrying its original metadata (e.g. source filename, party label, CSV columns).
2. At query time, the retriever fetches the top-k most relevant chunks for each input text.
3. Those chunks are injected as “Context:” system messages before your user prompt, grounding the LLM’s answer in your corpus.
4. This both improves factuality and lets you trace exactly which snippets (and their metadata) influenced each response.

The functions included in the Async-RAG part of the package let you simply build your own retriever database and then include in the `run_analysis()` pipeline:
1. **`build_retriever()`**: A synchronous helper that ingests your PDFs and/or CSVs, splits them into overlapping chunks (either fixed‐size or sentence‐aware), embeds each chunk via Ollama (or any provided embedder), and builds a FAISS or Chroma vector index.

2. **`build_retriever_async()`**: An async version of the above—ideal in Jupyter or any `async` workflow.  It batches your texts into large embed requests (configurable via `batch_size`) and throttles concurrency (`max_concurrency`) to maximize throughput while avoiding overload.

3. **RAG Integration in `run_analysis()`**:  You can now pass one—or a list—of retrievers directly into `run_analysis(...)`. 

