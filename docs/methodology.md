# Methodology

## Study sample

The submitted workflow cleans positive-price, geocoded rental listings and
retains observations within five miles (8.047 km) of an LA Metro station.
Listing-to-station distances use the haversine formula. Transit proximity is
represented continuously, with a 0.5-mile TOD indicator, and with four distance
bands: 0–0.5, 0.5–1, 1–2, and 2–5 miles.

## Composite Gentrification Risk Score

For each listing, the analysis constructs:

- **Rent premium (RP):** listing rent relative to 2024 ZCTA median gross rent.
- **Rent burden (RB):** listing rent relative to 30% of monthly ZCTA median income.
- **Short-run rent appreciation (SRA):** percentage change in ZCTA median rent from 2019 to 2024.
- **Price per square foot (PSP):** listing price per square foot, median-imputed when missing.
- **Education change (EDC):** change in the ZCTA share of adults 25+ with a bachelor's degree or higher.

Each component is standardized across the analytic sample. CGRS is their
unweighted mean; higher values indicate greater measured pressure under this
project's definition.

## Statistical comparison

A one-sided Mann–Whitney U test compares CGRS among listings inside the
0.5-mile TOD buffer with all retained listings outside that buffer. The test is
associational and does not control for neighborhood or listing composition.

## Prediction models

The primary training subset contains listings whose nearest station is on the
A or E Line. Predictors include:

- nearest-station and D Line scenario distances;
- TOD status and ordered proximity band;
- bedrooms, square feet, year built, and month;
- 2024 median rent, median income, poverty rate, and education share; and
- one-hot property-subtype indicators.

The analysis compares ordinary least squares, ridge regression, random forest,
and XGBoost using shuffled five-fold cross-validation with `random_state=42`.
Tree ensembles use 300 estimators. A second model comparison uses all lines as
a robustness check.

## Counterfactual scenario

For listings within five kilometers of one of four project-defined D Line
extension points, the submitted notebook sets `dist_to_olympics` to 0.3 km and
`in_tod_buffer` to 1 while holding other features fixed. The scenario is scored
with the fitted A/E-line random forest. It is a model-based perturbation, not a
causal or equilibrium forecast.
