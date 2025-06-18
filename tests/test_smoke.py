def test_import():
    import async_run_ollama as pl
    assert hasattr(pl, "run_analysis")
