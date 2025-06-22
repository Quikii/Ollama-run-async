def test_import():
    import async_run_ollama  as pl
    assert hasattr(pl, "run_analysis"), "run_analysis not found in async_run_ollama"

    import ollama_rag_run as rag
    # you can check for both sync & async builders:
    assert hasattr(rag, "build_retriever"),     "build_retriever not found in ollama_rag_run"
    assert hasattr(rag, "build_retriever_async"), "build_retriever_async not found in ollama_rag_run"

