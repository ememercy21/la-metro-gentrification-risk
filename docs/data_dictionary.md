# Data dictionary

## Input fields

| Variable | Source | Definition |
|---|---|---|
| `Zip` | Housing listings | Five-digit ZIP used to join ZCTA estimates |
| `latitude`, `longitude` | Housing listings | Geocoded listing coordinates |
| `LP` | Housing listings | Monthly listing price/rent used by the submitted analysis |
| `OLP` | Housing listings | Original listing price |
| `LP $/SqFt` | Housing listings | Listing price per square foot |
| `BR` | Housing listings | Number of bedrooms |
| `SqFt` | Housing listings | Interior area in square feet |
| `YB` | Housing listings | Year built |
| `Prop Subtype` | Housing listings | Property subtype |
| `file_date` | Housing listings | Snapshot date in `YYYYMMDD` form |
| `stop_id`, `stop_name` | LA Metro | Station identifier and name |
| `stop_lat`, `stop_lon` | LA Metro | Station coordinates |
| `Metro Line` | LA Metro | Line label attached to a station |
| `B25064_001E` | ACS | Median gross rent |
| `B19013_001E` | ACS | Median household income |
| `B15003_001E` | ACS | Population age 25+ used as education denominator |
| `B15003_022E`–`025E` | ACS | Bachelor's through doctorate counts |
| `B17001_001E`, `002E` | ACS | Poverty universe and population below poverty |

## Derived fields

| Variable | Definition |
|---|---|
| `dist_nearest_station` | Haversine distance in kilometers to the nearest station |
| `dist_to_olympics` | Distance in kilometers to the nearest added D Line scenario station |
| `proximity_band` | Nearest-station distance grouped as 0–0.5, 0.5–1, 1–2, or 2–5 miles |
| `in_tod_buffer` | 1 when a listing is within 0.5 miles of its nearest station |
| `RP` | $(LP-\text{median rent})/\text{median rent}$ |
| `RB` | $LP/(\text{median annual income}/12\times0.30)$ |
| `SRA` | Change in ACS median rent from 2019 to 2024 |
| `PSP_raw` | Listing price per square foot with median imputation |
| `EDC` | Change in bachelor's-or-higher share from 2019 to 2024 |
| `CGRS` | Unweighted mean of the five standardized components above |
| `LTGS` | Mean of standardized line-level median CGRS, mean SRA, and mean EDC |
| `CGRS_uplift` | Scenario prediction minus baseline prediction for one listing |

## Privacy boundary

The original listing table also contains exact addresses and agent contact
information. Those columns are not required for the model and the file is not
distributed in this repository.
