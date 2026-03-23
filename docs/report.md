# Benchmark Report
**Date:** 2026-03-23 | **Model:** sarvam-1-f16.gguf (llama 3B, 2.53B params, 5.05GB) | **CPU:** Intel i7-8650U | **Turbo:** OFF

## Setup

| | sarvam.c | llama.cpp |
|---|---|---|
| commit | `fa6f42c` | `380b4c9` (build 7376) |
| backend | CPU | CPU |
| kv cache | f16 | f16 |
| mmap | — | see per-run |

Turbo disabled via:
```bash
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
```

## Results

### sarvam.c :: 1 thread, no mmap (4 runs)
```
make benchmark "a" 200
```

| run | TBT (ms) | tok/s | total (ms) |
|-----|----------|-------|------------|
| 1 | 862.653 | 1.159 | 168217.243 |
| 2 | 854.498 | 1.170 | 166627.104 |
| 3 | 852.501 | 1.173 | 166237.714 |
| 4 | 849.989 | 1.176 | 165747.795 |
| **avg** | **854.9** | **1.170** | |

### llama.cpp :: 1 thread, no mmap (4 runs)
```
llama-bench -m sarvam-1-f16.gguf -p 0 -n 200 -t 1 --mmap 0
```

| run | TBT (ms) | tok/s | stddev tok/s |
|-----|----------|-------|--------------|
| 1 | 515.3 | 1.941 | 0.0119 |
| 2 | 504.3 | 1.983 | 0.0086 |
| 3 | 507.6 | 1.970 | 0.0149 |
| 4 | 513.3 | 1.948 | 0.0143 |
| **avg** | **510.1** | **1.961** | |

### llama.cpp :: 4 threads, mmap (4 runs)
```
llama-bench -m sarvam-1-f16.gguf -p 0 -n 200 -t 4 --mmap 1
```

| run | TBT (ms) | tok/s | stddev tok/s |
|-----|----------|-------|--------------|
| 1 | 288.3 | 3.470 | 0.071 |
| 2 | 295.9 | 3.382 | 0.124 |
| 3 | 306.7 | 3.265 | 0.134 |
| 4 | 299.4 | 3.356 | 0.272 |
| **avg** | **297.6** | **3.368** | |

> runs 3–4 show higher stddev, likely thermal throttling

## Summary

| config | tok/s | vs sarvam.c |
|---|---|---|
| sarvam.c (1t, no mmap, f16) | 1.170 | baseline |
| llama.cpp (1t, no mmap, f16) | 1.961 | 1.68x |
| llama.cpp (4t, mmap, f16) | 3.368 | 2.88x |