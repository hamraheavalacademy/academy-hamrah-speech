"""
Profile RTF, wall time, and CPU time for librosa.
Output is a torch.Tensor of shape [1, Time_Steps] so results are directly
comparable across all package scripts.
For memory profiling, run this same script under memray -- see commands below.

Requirements:
    pip install librosa torch numpy

Usage:
    # Timing / RTF (clean, no instrumentation overhead):
    python profile_librosa.py audio.wav 50

    # Memory (run separately, so it doesn't skew the timing numbers above):
    pip install memray
    memray run -o librosa_output.bin profile_librosa.py audio.wav 50
    memray stats librosa_output.bin
    # or for an interactive flamegraph:
    memray flamegraph librosa_output.bin
"""

import sys
import os
import time
import numpy as np
import psutil

TARGET_SR = 16000
DEFAULT_N_RUNS = 50


def load(path):
    import librosa
    import torch
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return torch.from_numpy(y).unsqueeze(0)  # [1, Time_Steps]


def get_audio_duration(path):
    import librosa
    # librosa.get_duration's kwarg name changed across versions (filename -> path)
    try:
        return librosa.get_duration(path=path)
    except TypeError:
        return librosa.get_duration(filename=path)


def profile_timing(path, n_runs):
    process = psutil.Process(os.getpid())

    # Untimed warm-up: absorbs import cost, disk cache warm-up, lazy backend
    # init -- none of that should count toward steady-state RTF/CPU numbers.
    load(path)

    wall_times, cpu_times = [], []
    for _ in range(n_runs):
        cpu_before = process.cpu_times()
        t0 = time.perf_counter()
        load(path)
        t1 = time.perf_counter()
        cpu_after = process.cpu_times()

        wall_times.append(t1 - t0)
        cpu_times.append(
            (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
        )

    return wall_times, cpu_times


def main():
    if len(sys.argv) < 2:
        print("Usage: python profile_librosa.py path/to/audio.wav [n_runs]")
        sys.exit(1)

    path = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_RUNS

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    audio_duration = get_audio_duration(path)
    print(f"Package: librosa")
    print(f"Audio file: {path}")
    print(f"Duration: {audio_duration:.3f}s | Runs: {n_runs} (+1 discarded warm-up)")

    sample_tensor = load(path)
    print(f"Output tensor: shape={tuple(sample_tensor.shape)}, dtype={sample_tensor.dtype}")

    wall_times, cpu_times = profile_timing(path, n_runs)

    avg_wall = float(np.mean(wall_times))
    std_wall = float(np.std(wall_times))
    avg_cpu = float(np.mean(cpu_times))
    rtf = avg_wall / audio_duration
    cpu_pct = (avg_cpu / avg_wall * 100) if avg_wall > 0 else 0.0

    print("\n--- Results ---")
    print(f"RTF:              {rtf:.4f}")
    print(f"Avg wall time:    {avg_wall:.4f}s (±{std_wall:.4f}s)")
    print(f"Avg CPU time:     {avg_cpu:.4f}s")
    print(f"CPU utilization:  {cpu_pct:.1f}%")
    print("\nFor peak memory, run under memray instead (see docstring at top of file).")


if __name__ == "__main__":
    main()
