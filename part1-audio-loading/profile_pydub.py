"""
Profile RTF, wall time, and CPU time for pydub.
Output is a torch.Tensor of shape [1, Time_Steps] so results are directly
comparable across all package scripts.
For memory profiling, run this same script under memray -- see commands below.

Requirements:
    pip install pydub torch numpy
    ffmpeg must be installed on the system (apt install ffmpeg / brew install ffmpeg)

Note: pydub shells out to ffmpeg as a subprocess, so wall time here includes
process-spawn + I/O wait, not just in-process compute. Compare CPU% against
wall time to see this gap.

Usage:
    # Timing / RTF (clean, no instrumentation overhead):
    python profile_pydub.py audio.wav 50

    # Memory (run separately, so it doesn't skew the timing numbers above):
    pip install memray
    memray run -o pydub_output.bin profile_pydub.py audio.wav 50
    memray stats pydub_output.bin
    # or for an interactive flamegraph:
    memray flamegraph pydub_output.bin
    # note: memray only tracks the Python process memory, not the separate
    # ffmpeg subprocess pydub spawns -- for that, watch ffmpeg's own RSS
    # separately (e.g. with psutil.Process(ffmpeg_pid) or `ps`).
"""

import sys
import os
import time
import numpy as np
import psutil

TARGET_SR = 16000
DEFAULT_N_RUNS = 50


def load(path):
    from pydub import AudioSegment
    import torch

    audio = AudioSegment.from_file(path).set_frame_rate(TARGET_SR).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    return torch.from_numpy(samples).unsqueeze(0)  # [1, Time_Steps]


def get_audio_duration(path):
    from pydub import AudioSegment
    audio = AudioSegment.from_file(path)
    return len(audio) / 1000.0  # pydub reports duration in milliseconds


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
        print("Usage: python profile_pydub.py path/to/audio.wav [n_runs]")
        sys.exit(1)

    path = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_RUNS

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    audio_duration = get_audio_duration(path)
    print(f"Package: pydub")
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
