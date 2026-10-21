# Results

`results.csv` is the row-level output from the latest experiment. It starts as a schema-only template. `comparison_table.csv` starts with metric columns and `NA` values because no results are claimed before running the code. Run `python run_experiment.py --top-k 3` to populate them.

After running, manually fill `answer_correct` and `hallucination` in `results.csv`, then run `python run_experiment.py --evaluate-only --top-k 3`. Save a copy of the reviewed results before rerunning the full experiment, since the full run overwrites `results.csv`.
