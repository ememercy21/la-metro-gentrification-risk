# Reproducibility notes

This repository keeps the submitted analysis intact where results depend on it
and makes only portability, privacy, and packaging changes.

## Changes made for GitHub

- Replaced the original absolute macOS path with repository-relative paths.
- Moved ACS cache files to `data/cache/` and output images to `outputs/figures/`.
- Removed notebook cell outputs and execution counts so local paths and stale
  state are not published.
- Exported notebook code to `src/analysis.py` for non-interactive execution.
- Excluded the raw housing table because it contains exact addresses and agent
  phone numbers and its redistribution rights are unclear.
- Removed drafts, nested ZIP archives, and LaTeX build debris from the public
  package; the final artifacts and source remain.

## Known analytical discrepancies

### Forecast model label

The report describes the D Line counterfactual as an XGBoost forecast, and
XGBoost has the lowest A/E-line cross-validated RMSE. In notebook cell 41,
however, both baseline and post-opening predictions are produced by `rf_ae`,
the fitted random forest. The repository preserves that implementation and
labels it explicitly.

### Two “median shift” statistics

The notebook reports two different summaries:

- the median of listing-level differences, `median(post - baseline) = -0.0023`;
- the difference between scenario medians, `median(post) - median(baseline) = -0.0132`.

The article's “median uplift” of approximately -0.013 refers to the second
quantity. These statistics need not be equal because the median is nonlinear.

## Interpretation and validation limits

1. **Target leakage / reconstruction.** Median rent and education variables are
   predictors and also contribute to CGRS construction. Model performance is
   therefore not an independent validation of future gentrification.
2. **Spatial dependence.** Random folds allow nearby listings from the same
   neighborhoods into training and validation sets, which can make error
   estimates optimistic.
3. **Temporal dependence.** The workflow models listing snapshots rather than a
   held-out future period.
4. **Sample-dependent standardization.** CGRS component z-scores use the full
   analytic sample, so values depend on the observations included.
5. **Scenario assumptions.** The counterfactual changes only two proximity
   features and ignores housing supply, displacement, zoning, macroeconomic
   conditions, and endogenous responses.
6. **Planned-station inputs.** Four station points are hard-coded in the
   notebook. Verify names, locations, and project status before reuse.

## Reference outputs

The small CSV files in `results/reference/` are transcribed from the submitted
notebook run. They are included for review and regression checking, not as a
substitute for rerunning the workflow on authorized source data.
