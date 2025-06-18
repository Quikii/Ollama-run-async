def test_import():
    import async_run_ollama.py as pl
    assert hasattr(pl, "run_analysis")
