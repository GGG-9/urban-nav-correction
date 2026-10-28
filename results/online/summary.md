# Online variant: learned correction of the causal filter

GNSS-outage 3-D position error, mean / max in metres (8 simulated test routes).

| route | B1 DR | B2 ESKF | Ours (B2 + corrector) |
|---|---|---|---|
| 06 | 118.9 / 346 | 29.6 / 83 | **29.64 / 82.9** |
| 07 | 113.8 / 397 | 66.2 / 201 | **66.17 / 201.3** |
| 42 | 85.5 / 259 | 56.2 / 301 | **56.18 / 301.3** |
| 43 | 137.0 / 413 | 46.7 / 174 | **46.74 / 174.2** |
| 44 | 112.6 / 385 | 27.9 / 90 | **27.94 / 90.0** |
| 45 | 75.6 / 237 | 64.4 / 265 | **64.40 / 264.6** |
| 46 | 59.0 / 208 | 50.0 / 125 | **50.05 / 125.3** |
| 47 | 133.9 / 376 | 52.4 / 172 | **52.36 / 171.8** |
| **test mean** | - | **49.18** / 301 | **49.18** / 301 |

Shrinkage coefficient fitted on the validation routes: alpha = 0.0; the base column is B2, the causal filter, not the smoother.

## Headline

* outage mean, causal filter -> corrected: **49.18 -> 49.18 m (+0.0 %)**
* worst single outage: **301 -> 301.3 m (+0.0 %)**
* routes evaluated: 8/8
