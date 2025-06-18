def test_import():
    import ollama_run_async.py as pl
    assert hasattr(pl, "run_analysis")
