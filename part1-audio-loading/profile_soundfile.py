"""
Profile RTF, wall time, and CPU time for soundfile.
Output is a torch.Tensor of shape [1, Time_Steps] so results are directly
comparable across all package scripts.

Resampling is done with plain numpy.interp (linear interpolation) rather than
another audio package's resampler, since soundfile itself has no resampler --
numpy/torch are the shared "output format" layer, not part of the comparison.

For memory profiling, run this same script under memray -- see commands below.

Requirements:
    pip install soundfile torch numpy

Usage:
    # Timing / RTF (clean, no instrumentation overhead):
    python profile_soundfile.py audio.wav 50

    # Memory (run separately, so it doesn't skew the timing numbers above):
    pip install memray
    memray run -o soundfile_output.bin profile_soundfile.py audio.wav 50
    memray stats soundfile_output.bin
    # or for an interactive flamegraph:
    memray flamegraph soundfile_output.bin
"""

import sys
import os
import time
import numpy as np
import psutil

TARGET_SR = 16000
DEFAULT_N_RUNS = 50


def load(path):
    import soundfile as sf
    import torch

    data, orig_sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if orig_sr != TARGET_SR:
        target_length = int(len(data) * TARGET_SR / orig_sr)
        original_indices = np.linspace(0, len(data) - 1, num=len(data))
        target_indices = np.linspace(0, len(data) - 1, num=target_length)
        data = np.interp(target_indices, original_indices, data)

    return torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # [1, Time_Steps]


def get_audio_duration(path):
    import soundfile as sf
    info = sf.info(path)
    return info.frames / info.samplerate


def profile_timing(path, n_runs):
    process = psutil.Process(os.getpid())

    load(path)  # untimed warm-up

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
        print("Usage: python profile_soundfile.py path/to/audio.wav [n_runs]")
        sys.exit(1)

    path = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_RUNS

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    audio_duration = get_audio_duration(path)
    print(f"Package: soundfile")
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
