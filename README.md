# Election Outliers

## Purpose and Use

The purpose of the election audit management tool "Election Outliers" (EO) is to increase public confidence in the quality of reported election results. EO is supposed to be used after the target election results are reported, but not before their availability. EO rank orders granular localities (precincts) based on predictability of their reported election results. The predictions are made with Lasso models that use various election results and voter registration data as powerful predictors. The predictability is measured with hypergeometric distribution functions applied to both models' values (as proxies for true values) and reported values. The most unpredictable precincts (outliers) are accumulated at the end of the ordered sequence. This ordered sequence is used to produce cumulative curves with the election results across election choices on the same chart. The right tail of each election choice curve answers the following two important questions:

1. Are outlier precincts biased against or in favor of this particular election choice? If so, an audit of these outlier precincts is advisable.
2. Do outlier precincts have an impact on the rank order among election choices? If so, an audit of these outlier precincts is imperative.

The bias of outlier precincts is visually detectable by a "structural break" in any curve's right tail, which exhibits either convexity or concavity relative to the rest of the curve. In other words, when the slope of the curve is abruptly changed at approximately 95% to 99% of the vote tally, the remaining 1% to 5% of the vote tally, as represented with the outlier precincts, may need to be audited. EO is very flexible regarding Lasso models' configurations, but simple choices for the models are strongly advisable. Fortunately, Lasso model type allows regularization-based simplification of unnecessarily complex user-defined model.

## Quickstart Guide

1. Obtain granular (precinct-level) election results, preferably directly from the states (the respective Secretary of State). Optionally, obtain also the benchmark election results from the same state.
2. Transform election results into the required format for "input_data_csv/" (e.g. USA_MS_20140624_RUN_USS_REP_000_data.csv and USA_MS_20201103_GEN_ALL_ALL_001_data.csv).
3. Prepare configuration for the job and its models, as defined in "input_jobs_json/" (e.g. USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000.json).
4. Create the batch of jobs (or one job) definition in "input_batches_csv/" (e.g. USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000_batch.csv).
5. Create the batch script in "input_batches_bat/" (e.g. USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000_batch.bat).
6. Run the batch script in "input_batches_bat/" or later in "output_jobs_dir/" (e.g. USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000_batch.bat)
7. Observe the output directory created or updated in "output_jobs_dir/" (e.g. USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000/)
8. Review the newly generated "PDF" and "SVG" files in "output_jobs_dir/<job_name>" (e.g <job_name> set to USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000, and files such as USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000_full_plots_parm.pdf, USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000_full_plots_parm.svg, USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000_magn_plots_parm.pdf, and USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000_magn_plots_parm.svg).
9. Review outlier precincts' names in the sorted table with outlier scores in "output_jobs_dir/<job_name>/<job_name>_data_parm.csv" (e.g. USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000_data_parm.csv).
10. Consider changing various configuration parameters in the configuration file from "input_jobs_json/" (e.g. USA_MS_20140624_RUN_USS_REP_TLY_BOTH_000.json), as well as "IS_AUDITED" flag (to 1) in data files from "input_data_csv/" (e.g. USA_MS_20140624_RUN_USS_REP_000_data.csv and USA_MS_20201103_GEN_ALL_ALL_001_data.csv) for research purposes and for obtaining additional insights. Additional output files may be generated after changes in job configuration file from "input_jobs_json/".
