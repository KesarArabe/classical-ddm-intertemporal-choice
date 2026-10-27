# Data

## Source

The project uses the ITC Database:

- Pongratz, H., & Schoemann, M. *A large-scale dataset of choice and
  response-time data in intertemporal choice*. Preprint / manuscript v2.0.
- OSF project: <https://osf.io/3wsae/>
- Direct file: `data/all_data.csv` in the OSF project.
- Licence: **CC BY-NC-SA 4.0**.
- The descriptor states that the dataset is a secondary dataset compiled from
  100 studies from 87 publications, and that the original authors were asked
  for permission to republish under this licence.

## Local file

The analysis reads the existing file:

```text
../data/itc_all_data.csv
```

relative to this project directory (i.e. `Default Project/data/itc_all_data.csv`).
It is not copied into this project and is never modified. To point the project
at a copy stored elsewhere, set `DDM_ITC_CSV` to its absolute path; the
environment variable takes precedence over the relative default.

Verified properties:

| Property | Value |
|---|---|
| Size | 484,996,376 bytes |
| SHA-256 | `14e5bf2f6434d791a5f68d8f0e4cec41e4c6bb671ce6fd80e83879f9759e0517` |
| Rows | 1,172,644 |
| Columns | 42 |
| Studies | 100 |
| Unique subjects | 11,852 |

The hash matches the OSF `data/all_data.csv` file (version 2, modified
2026-05-13). The local filename differs (`itc_all_data.csv`), but the contents
match.

## Downloading a fresh copy

If the file needs to be obtained again:

1. Open <https://osf.io/3wsae/>.
2. Download `data/all_data.csv`.
3. Verify the size and SHA-256 against the table above.
4. Place it at `../data/itc_all_data.csv`, or set the environment variable
   `DDM_ITC_CSV` to its absolute path.

## Licence and redistribution

CC BY-NC-SA 4.0 permits non-commercial use, requires attribution, and requires
share-alike terms for adapted material. The raw CSV is excluded from version
control by `.gitignore`. Do not commit it to a public repository.

## Citation

Use the dataset descriptor and the OSF project. The descriptor is a preprint,
not a peer-reviewed article; do not describe it as peer-reviewed. Cite the
original studies as well when reporting study-specific results.
