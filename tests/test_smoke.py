def test_import():
    import parallel_llama_df_analysis_v2 as pl
    assert hasattr(pl, "run_analysis")
