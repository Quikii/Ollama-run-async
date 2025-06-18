def test_import():
    import parallel_llama_df_analysis as pl
    assert hasattr(pl, "run_analysis")
