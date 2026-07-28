"""Profile MFCC extraction with Librosa.

All Part 2 scripts return a torch.Tensor of shape [1, 13, Frames] and use the
same sample rate, frame/hop lengths, FFT size, Mel filterbank, logarithm, and
DCT convention so their outputs and timing results are directly comparable.
Audio loading/resampling is performed once before the timed runs.

Requirements:
    pip install librosa soundfile scipy torch numpy psutil memray

Usage:
    python profile_librosa.py ../data/test1.wav 50
    memray run -o librosa_output.bin profile_librosa.py ../data/test1.wav 50
    memray stats librosa_output.bin
    memray flamegraph librosa_output.bin
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16_000
FRAME_LENGTH = 400
HOP_LENGTH = 160
N_FFT = 512
N_MELS = 40
N_MFCC = 13
DEFAULT_N_RUNS = 50


def load_audio(path):
    """Load mono float32 audio and resample to TARGET_SR outside timed runs."""
    signal, sample_rate = sf.read(path, always_2d=False)
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sample_rate != TARGET_SR:
        from math import gcd
        divisor = gcd(sample_rate, TARGET_SR)
        signal = resample_poly(signal, TARGET_SR // divisor, sample_rate // divisor)
    if signal.size < N_FFT:
        signal = np.pad(signal, (0, N_FFT - signal.size))
    return np.ascontiguousarray(signal, dtype=np.float32)


def get_audio_duration(path):
    info = sf.info(path)
    return info.frames / info.samplerate


def hz_to_mel_htk(frequencies_hz):
    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + frequencies_hz / 700.0)


def mel_to_hz_htk(mel_values):
    mel_values = np.asarray(mel_values, dtype=np.float64)
    return 700.0 * (10.0 ** (mel_values / 2595.0) - 1.0)


def build_shared_mel_filterbank(sample_rate):
    """Build the same HTK triangular Mel filterbank for all backends."""
    fft_frequencies = np.linspace(0.0, sample_rate / 2, N_FFT // 2 + 1)
    mel_edges = np.linspace(
        hz_to_mel_htk(20.0), hz_to_mel_htk(sample_rate / 2), N_MELS + 2
    )
    hz_edges = mel_to_hz_htk(mel_edges)
    filterbank = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float64)
    eps = np.finfo(float).eps
    for band in range(N_MELS):
        lower, center, upper = hz_edges[band:band + 3]
        lower_slope = (fft_frequencies - lower) / max(center - lower, eps)
        upper_slope = (upper - fft_frequencies) / max(upper - center, eps)
        filterbank[band] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    return filterbank.astype(np.float32)
PACKAGE_NAME = "librosa"


def extract_mfcc(signal):
    import librosa
    import torch

    stft_matrix = librosa.stft(
        y=signal,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=FRAME_LENGTH,
        window="hamming",
        center=False,
    )
    power = (np.abs(stft_matrix) ** 2).astype(np.float32)
    mel_power = build_shared_mel_filterbank(TARGET_SR) @ power
    mel_db = librosa.power_to_db(mel_power, ref=1.0, amin=1e-10, top_db=None)
    mfcc = librosa.feature.mfcc(
        S=mel_db, n_mfcc=N_MFCC, dct_type=2, norm="ortho", lifter=0
    )
    return torch.from_numpy(np.asarray(mfcc, dtype=np.float32)).unsqueeze(0)

def profile_timing(signal, n_runs):
    process = psutil.Process(os.getpid())
    extract_mfcc(signal)  # untimed warm-up

    wall_times, cpu_times = [], []
    sample_output = None
    for _ in range(n_runs):
        cpu_before = process.cpu_times()
        t0 = time.perf_counter()
        sample_output = extract_mfcc(signal)
        t1 = time.perf_counter()
        cpu_after = process.cpu_times()

        wall_times.append(t1 - t0)
        cpu_times.append(
            (cpu_after.user - cpu_before.user)
            + (cpu_after.system - cpu_before.system)
        )
    return sample_output, wall_times, cpu_times


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} path/to/audio.wav [n_runs]")
        raise SystemExit(1)

    path = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N_RUNS

    if not os.path.exists(path):
        print(f"File not found: {path}")
        raise SystemExit(1)
    if n_runs < 1:
        print("n_runs must be at least 1")
        raise SystemExit(1)

    signal = load_audio(path)
    audio_duration = get_audio_duration(path)
    output, wall_times, cpu_times = profile_timing(signal, n_runs)

    avg_wall = float(np.mean(wall_times))
    std_wall = float(np.std(wall_times))
    avg_cpu = float(np.mean(cpu_times))
    rtf = avg_wall / audio_duration if audio_duration > 0 else float("nan")
    cpu_pct = (avg_cpu / avg_wall * 100.0) if avg_wall > 0 else 0.0

    print(f"Package: {PACKAGE_NAME}")
    print(f"Audio file: {path}")
    print(f"Duration: {audio_duration:.3f}s | Runs: {n_runs} (+1 discarded warm-up)")
    print(f"Output tensor: shape={tuple(output.shape)}, dtype={output.dtype}")
    print("\n--- Results ---")
    print(f"RTF:              {rtf:.4f}")
    print(f"Avg wall time:    {avg_wall:.4f}s (±{std_wall:.4f}s)")
    print(f"Avg CPU time:     {avg_cpu:.4f}s")
    print(f"CPU utilization:  {cpu_pct:.1f}%")
    print("\nFor peak memory, run this script under memray (see its docstring).")


if __name__ == "__main__":
    main()
