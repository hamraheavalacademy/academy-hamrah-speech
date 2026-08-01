# Part 2 — MFCC Feature Extraction

This folder contains three comparable MFCC implementations and one standalone
comparison/profiling runner.

## Files

- `profile_librosa.py`
- `profile_scipy.py`
- `profile_pytorch.py`
- `benchmark_feature_extraction.py`

The previous Jupyter notebook has been replaced by
`benchmark_feature_extraction.py`, so the complete comparison can be executed
from a normal Python process and profiled with Memray.

All MFCC implementations:

- return a `torch.Tensor` with shape `[1, 13, Frames]`;
- use a 16 kHz target sample rate;
- use a 25 ms frame (`400` samples) and a 10 ms hop (`160` samples);
- use `N_FFT=512`, `40` HTK Mel filters, log power in dB, and orthonormal DCT-II;
- load and resample audio once before the timed runs;
- perform one untimed warm-up.

## Install requirements

```bash
pip install librosa soundfile scipy torch numpy psutil memray
```

## Run an individual backend

From the `part2-feature-extraction` directory:

```bash
python profile_librosa.py ../data/test1.wav 50
python profile_scipy.py ../data/test1.wav 50
python profile_pytorch.py ../data/test1.wav 50
```

PyTorch runs on CPU by default for a fair comparison.

## Run the complete comparison

One or more audio files can be supplied:

```bash
python benchmark_feature_extraction.py \
    ../data/test1.wav ../data/test2.wav --runs 50
```

The script reports:

1. output shape and dtype;
2. MAE, RMSE, and maximum absolute error relative to Librosa;
3. average wall time and standard deviation;
4. average process CPU time and CPU utilization;
5. Real-Time Factor (RTF).

Process CPU time is measured with Python’s high-resolution `time.process_time()`
around the complete block of timed runs and then divided by the run count. This is
more reliable for very fast functions than sampling a coarse process timer around
every individual call.

To benchmark only one backend:

```bash
python benchmark_feature_extraction.py \
    ../data/test1.wav --backend scipy --runs 50
```

To save the tables as machine-readable JSON:

```bash
python benchmark_feature_extraction.py \
    ../data/test1.wav --runs 50 --json-output results.json
```

## Memory profiling with Memray

Profile one implementation at a time. `--skip-accuracy` prevents unrelated
reference calculations from entering the memory profile.

```bash
memray run -o librosa_output.bin benchmark_feature_extraction.py \
    ../data/test1.wav --backend librosa --runs 50 --skip-accuracy

memray run -o scipy_output.bin benchmark_feature_extraction.py \
    ../data/test1.wav --backend scipy --runs 50 --skip-accuracy

memray run -o pytorch_output.bin benchmark_feature_extraction.py \
    ../data/test1.wav --backend pytorch --runs 50 --skip-accuracy
```

Inspect a profile with:

```bash
memray stats scipy_output.bin
memray flamegraph scipy_output.bin
```

The original individual scripts can also be run directly under Memray, as shown
in their module docstrings.
