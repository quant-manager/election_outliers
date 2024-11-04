# Election Outliers

## Purpose and Use

The purpose of the election audit management tool "Election Outliers" (EO) is to increase public confidence in the quality of reported election results. EO is supposed to be used after the target election results are reported, but not before their availability. EO rank orders granular localities (precincts) based on predictability of their reported election results. The predictions are made with Lasso models that use various election results and voter registration data as powerful predictors. The predictability is measured with hypergeometric distribution functions applied to both models' values (as proxies for true values) and reported values. The most unpredictable precincts (outliers) are accumulated at the end of the ordered sequence. This ordered sequence is used to produce cumulative curves with the election results across election choices on the same chart. The right tail of each election choice curve answers the following two important questions:

1. Are outlier precincts biased against or in favor of this particular election choice? If so, an audit of these outlier precincts is advisable.
2. Do outlier precincts have an impact on the rank order among election choices? If so, an audit of these outlier precincts is imperative.

The bias of outlier precincts is visually detectable by a "structural break" in any curve's right tail, which exhibits either convexity or concavity relative to the rest of the curve. In other words, when the slope of the curve is abruptly changed at approximately 95% to 99% of the vote tally, the remaining 1% to 5% of the vote tally, as represented with the outlier precincts, may need to be audited. EO is very flexible regarding Lasso models' configurations, but simple choices for the models are strongly advisable. Fortunately, Lasso model type allows regularization-based simplification of unnecessarily complex user-defined model.

## Quickstart Guide

...
