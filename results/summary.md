# Results summary

GNSS-outage 3-D position error, mean / max in metres (8 simulated test routes).

| route | B1 DR | B2 ESKF | B3 ESKF+RTS | GPU ESKF+RTS | GPU+map | Ours |
|---|---|---|---|---|---|---|
| 06 | 119.8 / 375 | 47.7 / 195 | 7.67 / 15.9 | 8.31 / 15.3 | 4.46 / 13.0 | 8.31 / 15.3 |
| 07 | 113.8 / 397 | 66.2 / 201 | 8.43 / 15.0 | 12.30 / 25.4 | 8.63 / 23.3 | 12.30 / 25.4 |
| 42 | 85.5 / 259 | 56.2 / 301 | 12.83 / 25.7 | 13.41 / 25.5 | 9.92 / 20.5 | 13.41 / 25.5 |
| 43 | 137.0 / 413 | 46.7 / 174 | 9.00 / 11.5 | 9.18 / 13.7 | 5.69 / 13.3 | 9.18 / 13.7 |
| 44 | 112.6 / 385 | 27.9 / 90 | 9.51 / 15.0 | 10.75 / 18.2 | 6.29 / 12.7 | 10.75 / 18.2 |
| 45 | 75.6 / 237 | 64.4 / 265 | 14.74 / 24.5 | 15.02 / 25.8 | 9.44 / 24.5 | 15.02 / 25.8 |
| 46 | 59.0 / 208 | 50.0 / 125 | 6.98 / 14.3 | 6.87 / 14.6 | 4.63 / 14.6 | 6.87 / 14.6 |
| 47 | 133.9 / 376 | 52.4 / 172 | 10.31 / 28.3 | 10.14 / 29.6 | 6.77 / 29.8 | 10.14 / 29.6 |
| **test mean** | **104.65** / 413 | **51.44** / 301 | **9.93** / 28 | **10.75** / 30 | **6.98** / 30 | **10.75** / 30 |

B1-B3 are the numpy reference in `runs/kf`. "Ours" refines the base trajectory `src.models.train` actually used - the windowed GPU output when `runs/kf_gpu` exists, otherwise the numpy B3. The shrinkage coefficient comes from a one-standard-error gate on the validation routes and is alpha = 0.0; at alpha = 0 the module runs in pass-through and the column repeats the base.

`GPU+map` is not a learned correction: the outage trajectory is marched along the road centreline under a continuity constraint (`scripts/map_prior.py`). The map is new information the filter never sees, which is why it works where the corrector does not; it removes the cross-track error and leaves the along-track part.

## Headline

* outage mean, dead reckoning -> ESKF+RTS: **104.6 -> 9.93 m (-90.5 %)**
* worst single outage, same step: **413 -> 28.3 m (-93.1 %)**
* window-parallel GPU pipeline vs numpy B3 on the same routes: **10.75 vs 9.93 m** (+8.2 %) - the cost of windowing: each window sees +-90 s of fixes instead of the whole route
* road map prior on the GPU trajectory: **10.75 -> 6.98 m (+35.1 %)** - the map removes the cross-track part of the outage error; the nearest-point projection alone gives less and regresses self-intersecting routes
* mapped lane line (fig8): the lateral offset of a lane line built from these poses is **43.9 m** for B1, 6.5 m for the GPU output and **0.28 m** with the map prior; the along-road part stays at 40.7 -> 6.9 -> 6.9 m - a map cannot remove it, and the map prior correctly does not try
* learned corrector (alpha = 0.0, 8/8 test routes): **held back by the one-SE validation gate** - the correction does not clear the bar on held-out routes, so the module runs in pass-through and Ours = the filtered trajectory (10.75 m); the gate lets it only improve on the delivered trajectory, never degrade it

## Throughput

* device: NVIDIA GeForce RTX 4090, window T = 2000, chunk = 128, median of 3
* 1,000,000 rows: CPU 2244.6 s vs GPU 46.21 s -> **49x**
