# Dataset

## `rul_steps.csv`

One row per weekly capacity test of every cell, from the start of cycling until the cell reaches its end of life (capacity below 80% of the 250 mAh rated capacity). 245 cells, 63 experimental groups, 3,956 rows.

| Column | Unit | Meaning |
| --- | --- | --- |
| `cell` | | Cell name, for example `G36C4` (group 36, cell 4) |
| `group` | | Experimental group. Cells in a group were cycled under the same conditions |
| `step` | | Index of the weekly test, starting at 0 |
| `capacity` | Ah | Full discharge capacity measured at this test (C/5) |
| `efc_so_far` | EFC | Equivalent full cycles used before this test: cumulative discharged Ah / 0.25 |
| `rul_efc` | EFC | Remaining useful life: equivalent full cycles left until end of life. **Prediction target** |
| `efc_total` | EFC | Total lifetime of the cell (`efc_so_far + rul_efc`) |
| `c_chg` | C | Charge rate: maximum charge current in the first 30 cycles / 0.25 A |
| `c_dis` | C | Discharge rate: median discharge current in the first 30 cycles / 0.25 A |
| `dod` | fraction | Depth of discharge: median discharged capacity of the first 10 cycles / first measured capacity |

The end of life is interpolated linearly between the two weekly tests around the 80% threshold.

## Source

This table is derived from the **ISU-ILCC Battery Aging Dataset** by `01_build_dataset.ipynb`. The raw data (about 11 GB) is not included in this repository.

- Download: https://doi.org/10.25380/iastate.22582234
- 251 lithium-polymer cells (NMC cathode, graphite anode, 250 mAh), cycled at 30 °C under 63 combinations of charge rate, discharge rate and depth of discharge
- Six cells are not in the table because they did not reach end of life or had missing records

## How the cells were tested

- 502030 lithium-polymer pouch cells (Shenzhen Honghaosheng Electronics), 3.0–4.2 V
- Two 64-channel Neware BTS-4000 cyclers (Iowa State University and Iowa Lakes Community College), cells in incubators at 30 °C
- Every cycle: constant-current, constant-voltage charge, then constant-current discharge to a voltage cutoff that sets the depth of discharge
- Every 7 days a reference performance test (one full C/5 cycle and one full C/2 cycle); groups 20–64 have an extra test after the first 3.5 days. `capacity` is the C/5 discharge capacity of these tests
- Groups 1–25 discharge at about 0.5C and vary the charge rate and depth of discharge; groups 26–64 vary all three factors
- License: CC BY 4.0

Citation: T. Li, Z. Zhou, A. Thelen, D. Howey, C. Hu. "Predicting Battery Lifetime Under Varying Usage Conditions from Early Aging Data." Cell Reports Physical Science, 2024. https://doi.org/10.48550/arXiv.2307.08382
