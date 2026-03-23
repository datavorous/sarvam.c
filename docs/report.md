# Benchmark Report
**Date:** 2026-03-23 | **Model:** sarvam-1-f16.gguf (llama 3B, 2.53B params, 5.05GB) | **CPU:** Intel i7-8650U | **Turbo:** OFF

## Setup

| | sarvam.c | llama.cpp |
|---|---|---|
| commit | `fa6f42c` | `380b4c9` (build 7376) |
| backend | CPU | CPU |
| kv cache | f16 | f16 |
| mmap | — | see per-run |

Compiler flags:
```bash
-O3 -ffast-math -march=native -funroll-loops -fomit-frame-pointer
```

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

## OpenBLAS `matmul` experiment
**commit:** `6a77a9a`

Replaced naive matmul loop with `cblas_sgemv`. 

### why?

<img src="https://private-user-images.githubusercontent.com/70896212/567573899-012730b6-05f6-4b15-97f5-756198343678.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzQyNzEwMDYsIm5iZiI6MTc3NDI3MDcwNiwicGF0aCI6Ii83MDg5NjIxMi81Njc1NzM4OTktMDEyNzMwYjYtMDVmNi00YjE1LTk3ZjUtNzU2MTk4MzQzNjc4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzIzVDEyNTgyNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTI0MjFiOTk0ZWJlNzY5MWQ1ZDhlYjIzYWQ4YWVmOWFjODkxZDgyMjJiMTFjNTM0MzAxYTM2ODQzZDkyOTQyZmUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.MJ_61Q9wDPCRNbOFN3w4y7qEjilJAsWTkKXT2Md3UMY">

<img src="https://private-user-images.githubusercontent.com/70896212/567575736-0c61e82f-11df-4e3b-8536-cacdf4985eb2.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzQyNzEwMDYsIm5iZiI6MTc3NDI3MDcwNiwicGF0aCI6Ii83MDg5NjIxMi81Njc1NzU3MzYtMGM2MWU4MmYtMTFkZi00ZTNiLTg1MzYtY2FjZGY0OTg1ZWIyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzIzVDEyNTgyNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTAyMjA5NjNkYWVlYjA2ZTY1MWE4YWFmZThmMGM5ODUwODg1YjNjYjA0NzA0M2VhZDVjN2MxZWQwZDg3YjkwY2YmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.t1whK0HZqTOWBtA2JOLo9ATKpPRMEsnG9kMfDcZ3KoM">

### results (6 runs, no turbo)

| run | TBT (ms) | tok/s |
|-----|----------|-------|
| 1 | 449.1 | 2.227 |
| 2 | 503.1 | 1.987 |
| 3 | 476.3 | 2.100 |
| 4 | 503.0 | 1.988 |
| 5 | 549.4 | 1.820 |

### finding
at 1 thread, sarvam.c + OpenBLAS ~ llama.cpp (1t, no mmap).

## full comparison

| config | tok/s | threads |
|---|---|---|
| sarvam.c naive | 1.17 | 1 |
| sarvam.c + OpenBLAS | ~2.02| 1 |
| llama.cpp (no mmap) | 1.96 | 1 |
| llama.cpp (4t, mmap) | 3.37 | 4 |