"""Simple MFCC comparison for Librosa, SciPy, and PyTorch.

Usage:
    python benchmark_feature_extraction_simple.py audio.wav 50 all
    python benchmark_feature_extraction_simple.py audio.wav 50 scipy

Memray:
    memray run -o scipy.bin benchmark_feature_extraction_simple.py audio.wav 50 scipy
    memray stats scipy.bin
    memray flamegraph scipy.bin
"""

import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np


BACKENDS = {
    "librosa": ("profile_librosa", "Librosa"),
    "scipy": ("profile_scipy", "SciPy/NumPy"),
    "pytorch": ("profile_pytorch", "PyTorch CPU"),
}


def load_selected_backends(choice):
    """Import all backends or only the selected one."""
    names = list(BACKENDS) if choice == "all" else [choice]
    return {
        name: importlib.import_module(BACKENDS[name][0])
        for name in names
    }


def compare_outputs(path, modules):
    """Check output shape and compare values with Librosa."""
    outputs = {}

    print("\n=== Output check ===")
    for name, module in modules.items():
        signal = module.load_audio(str(path))
        tensor = module.extract_mfcc(signal)
        outputs[name] = tensor.detach().cpu().numpy()

        print(
            f"{BACKENDS[name][1]:12s} "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
        )

    if "librosa" not in outputs or len(outputs) == 1:
        return

    print("\n=== Numerical difference from Librosa ===")
    reference = outputs["librosa"]

    for name, candidate in outputs.items():
        frames = min(reference.shape[-1], candidate.shape[-1])
        diff = candidate[..., :frames] - reference[..., :frames]

        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        max_error = np.max(np.abs(diff))

        print(
            f"{BACKENDS[name][1]:12s} "
            f"MAE={mae:.8f} | "
            f"RMSE={rmse:.8f} | "
            f"Max={max_error:.8f}"
        )


def benchmark_backend(path, name, module, runs):
    """Measure wall time, process CPU time, CPU use, and RTF."""
    signal = module.load_audio(str(path))
    duration = module.get_audio_duration(str(path))

    module.extract_mfcc(signal)  # Untimed warm-up

    wall_times = []
    output = None

    cpu_start = time.process_time()

    for _ in range(runs):
        start = time.perf_counter()
        output = module.extract_mfcc(signal)
        wall_times.append(time.perf_counter() - start)

    total_cpu = time.process_time() - cpu_start
    total_wall = sum(wall_times)

    avg_wall = np.mean(wall_times)
    std_wall = np.std(wall_times)
    avg_cpu = total_cpu / runs
    cpu_percent = 100.0 * total_cpu / total_wall if total_wall else 0.0
    rtf = avg_wall / duration

    return {
        "backend": BACKENDS[name][1],
        "wall_ms": avg_wall * 1000,
        "std_ms": std_wall * 1000,
        "cpu_ms": avg_cpu * 1000,
        "cpu_percent": cpu_percent,
        "rtf": rtf,
        "shape": tuple(output.shape),
    }


def print_results(results):
    """Print one compact benchmark table."""
    print("\n=== Runtime benchmark ===")
    print(
        f"{'Backend':12s} | {'Wall ms':>10s} | {'Std ms':>10s} | "
        f"{'CPU ms':>10s} | {'CPU %':>8s} | {'RTF':>10s}"
    )
    print("-" * 75)

    for item in results:
        print(
            f"{item['backend']:12s} | "
            f"{item['wall_ms']:10.4f} | "
            f"{item['std_ms']:10.4f} | "
            f"{item['cpu_ms']:10.4f} | "
            f"{item['cpu_percent']:7.1f}% | "
            f"{item['rtf']:10.6f}"
        )


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python benchmark_feature_extraction_simple.py "
            "audio.wav [runs] [all|librosa|scipy|pytorch]"
        )
        raise SystemExit(1)

    path = Path(sys.argv[1]).expanduser().resolve()
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    choice = sys.argv[3].lower() if len(sys.argv) > 3 else "all"

    if not path.is_file():
        raise SystemExit(f"Audio file not found: {path}")
    if runs < 1:
        raise SystemExit("runs must be at least 1")
    if choice not in {"all", *BACKENDS}:
        raise SystemExit("backend must be: all, librosa, scipy, or pytorch")

    # Keep the main comparison on CPU.
    os.environ["USE_CUDA"] = "0"

    modules = load_selected_backends(choice)

    # Numerical comparison is useful only when multiple backends are selected.
    if choice == "all":
        compare_outputs(path, modules)

    results = [
        benchmark_backend(path, name, module, runs)
        for name, module in modules.items()
    ]
    print_results(results)

    print("\nAudio loading and resampling were outside the timed region.")
    print("One untimed warm-up was performed for each backend.")


if __name__ == "__main__":
    main()
