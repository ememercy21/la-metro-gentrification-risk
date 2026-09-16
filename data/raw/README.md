# Data setup

## Included

`MetroLines_with_Lines.csv` contains LA Metro station identifiers, names,
coordinates, and line labels used by the submitted notebook. The project adds
four planned D Line extension points in code for its scenario analysis.

## Not included: housing listings

The submitted `housing_combined_data_geocoded.csv` is excluded from this public
repository because it contains exact street addresses and agent phone numbers,
and the underlying redistribution rights are unclear. Do not commit it.

If you have an authorized copy, place it here as:

```text
data/raw/housing_combined_data_geocoded.csv
```

The workflow expects these modeling fields:

| Field | Purpose |
|---|---|
| `Zip` | Join to ACS ZCTA estimates |
| `latitude`, `longitude` | Station-distance calculation |
| `LP`, `OLP` | Listing and original listing price |
| `LP $/SqFt` | Price-per-square-foot component |
| `BR`, `SqFt`, `YB` | Bedrooms, area, and year built |
| `Prop Subtype` | Property-type indicators |
| `file_date` | Observation month |

Other columns may exist in the source file, but personally identifying or
contact fields are not required for the analysis.

## ACS data

The notebook retrieves 2019 and 2024 ACS five-year ZCTA estimates from the
U.S. Census API. Add `CENSUS_API_KEY` to a local `.env` file. Responses are
cached in `data/cache/` and ignored by Git.
