"""Standalone comparison and profiling runner for Part 2 MFCC backends.

This script replaces the former Jupyter notebook. It verifies output shape and
numerical agreement, then benchmarks the selected backend(s) using the same
preloaded audio signal. Audio loading and resampling are outside the timed
region.

CPU time is measured once across the complete timed block and divided by the
number of runs. This avoids the coarse timer-resolution problem that can occur
when process CPU time is sampled around every very short MFCC call.

Requirements:
    pip install librosa soundfile scipy torch numpy psutil memray

Usage:
    python benchmark_feature_extraction.py ../data/test1.wav ../data/test2.wav --runs 50
    python benchmark_feature_extraction.py ../data/test1.wav --backend scipy --runs 50

Memray (profile one backend in one normal Python process):
    memray run -o scipy_benchmark.bin benchmark_feature_extraction.py \
        ../data/test1.wav --backend scipy --runs 50 --skip-accuracy
    memray stats scipy_benchmark.bin
    memray flamegraph scipy_benchmark.bin
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_N_RUNS = 50

BACKEND_FILES = {
    "librosa": "profile_librosa.py",
    "scipy": "profile_scipy.py",
    "pytorch": "profile_pytorch.py",
}
BACKEND_LABELS = {
    "librosa": "Librosa",
    "scipy": "SciPy/NumPy",
    "pytorch": "PyTorch CPU",
}


@dataclass(frozen=True)
class VerificationResult:
    file: str
    backend: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class AccuracyResult:
    file: str
    backend: str
    compared_frames: int
    mae: float
    rmse: float
    max_abs_error: float


@dataclass(frozen=True)
class BenchmarkResult:
    file: str
    backend: str
    runs: int
    avg_wall_ms: float
    std_wall_ms: float
    avg_cpu_ms: float
    cpu_utilization_percent: float
    rtf: float
    shape: tuple[int, ...]
    dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and benchmark the Part 2 MFCC implementations."
    )
    parser.add_argument(
        "audio",
        nargs="+",
        type=Path,
        help="One or more WAV files to benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help=f"Timed MFCC extractions per backend (default: {DEFAULT_N_RUNS}).",
    )
    parser.add_argument(
        "--backend",
        choices=["all", *BACKEND_FILES],
        default="all",
        help="Benchmark all backends or only one backend (default: all).",
    )
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip numerical comparison. Recommended when running under Memray.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optionally save all collected results as JSON.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[Path]:
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    paths: list[Path] = []
    for raw_path in args.audio:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Audio file not found: {path}")
        paths.append(path)
    return paths


def selected_backend_names(choice: str) -> list[str]:
    return list(BACKEND_FILES) if choice == "all" else [choice]


def load_backend(name: str) -> ModuleType:
    script_path = SCRIPT_DIR / BACKEND_FILES[name]
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing backend script: {script_path}")

    module_name = f"part2_{name}_backend"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load backend module: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def profile_backend(
    module: ModuleType,
    signal: np.ndarray,
    audio_duration: float,
    n_runs: int,
) -> tuple[object, list[float], float]:
    """Return output, per-run wall times, and average process CPU time.

    The warm-up is intentionally excluded. CPU time is sampled around the full
    run block, rather than around every call, for better precision on fast code.
    """
    module.extract_mfcc(signal)  # untimed warm-up

    wall_times: list[float] = []
    output = None

    cpu_before = time.process_time()
    for _ in range(n_runs):
        t0 = time.perf_counter()
        output = module.extract_mfcc(signal)
        wall_times.append(time.perf_counter() - t0)
    cpu_after = time.process_time()

    if output is None:
        raise RuntimeError("No benchmark output was produced")
    if audio_duration <= 0:
        raise ValueError("Audio duration must be positive")

    avg_cpu = max(0.0, cpu_after - cpu_before) / n_runs
    return output, wall_times, avg_cpu


def verify_outputs(
    audio_paths: Iterable[Path],
    modules: dict[str, ModuleType],
) -> tuple[list[VerificationResult], list[AccuracyResult]]:
    verification: list[VerificationResult] = []
    accuracy: list[AccuracyResult] = []

    for audio_path in audio_paths:
        outputs: dict[str, np.ndarray] = {}
        for name, module in modules.items():
            signal = module.load_audio(str(audio_path))
            tensor = module.extract_mfcc(signal)
            array = tensor.detach().cpu().numpy()
            outputs[name] = array
            verification.append(
                VerificationResult(
                    file=audio_path.name,
                    backend=BACKEND_LABELS[name],
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                )
            )

        if "librosa" not in outputs:
            continue

        reference = outputs["librosa"]
        for name, candidate in outputs.items():
            frames = min(reference.shape[-1], candidate.shape[-1])
            if frames < 1:
                raise RuntimeError(
                    f"No comparable MFCC frames for {audio_path.name} ({name})"
                )
            diff = candidate[..., :frames] - reference[..., :frames]
            accuracy.append(
                AccuracyResult(
                    file=audio_path.name,
                    backend=BACKEND_LABELS[name],
                    compared_frames=frames,
                    mae=float(np.mean(np.abs(diff))),
                    rmse=float(np.sqrt(np.mean(np.square(diff)))),
                    max_abs_error=float(np.max(np.abs(diff))),
                )
            )

    return verification, accuracy


def benchmark(
    audio_paths: Iterable[Path],
    modules: dict[str, ModuleType],
    n_runs: int,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for audio_path in audio_paths:
        for name, module in modules.items():
            signal = module.load_audio(str(audio_path))
            duration = float(module.get_audio_duration(str(audio_path)))
            output, wall_times, avg_cpu = profile_backend(
                module=module,
                signal=signal,
                audio_duration=duration,
                n_runs=n_runs,
            )

            total_wall = float(np.sum(wall_times))
            avg_wall = float(np.mean(wall_times))
            results.append(
                BenchmarkResult(
                    file=audio_path.name,
                    backend=BACKEND_LABELS[name],
                    runs=n_runs,
                    avg_wall_ms=avg_wall * 1000.0,
                    std_wall_ms=float(np.std(wall_times)) * 1000.0,
                    avg_cpu_ms=avg_cpu * 1000.0,
                    cpu_utilization_percent=(
                        (avg_cpu * n_runs) / total_wall * 100.0
                        if total_wall > 0
                        else 0.0
                    ),
                    rtf=avg_wall / duration,
                    shape=tuple(output.shape),
                    dtype=str(output.dtype),
                )
            )

    return results


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(no rows)"
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def print_verification(results: list[VerificationResult]) -> None:
    print("\n=== Output verification ===")
    rows = [
        [item.file, item.backend, str(item.shape), item.dtype]
        for item in results
    ]
    print(format_table(["file", "backend", "shape", "dtype"], rows))


def print_accuracy(results: list[AccuracyResult]) -> None:
    print("\n=== Numerical agreement (Librosa reference) ===")
    rows = [
        [
            item.file,
            item.backend,
            str(item.compared_frames),
            f"{item.mae:.8f}",
            f"{item.rmse:.8f}",
            f"{item.max_abs_error:.8f}",
        ]
        for item in results
    ]
    print(
        format_table(
            ["file", "backend", "frames", "MAE", "RMSE", "max abs error"],
            rows,
        )
    )


def print_benchmarks(results: list[BenchmarkResult]) -> None:
    print("\n=== Runtime and CPU benchmark ===")
    rows = [
        [
            item.file,
            item.backend,
            str(item.runs),
            f"{item.avg_wall_ms:.4f}",
            f"{item.std_wall_ms:.4f}",
            f"{item.avg_cpu_ms:.4f}",
            f"{item.cpu_utilization_percent:.1f}%",
            f"{item.rtf:.6f}",
        ]
        for item in results
    ]
    print(
        format_table(
            [
                "file",
                "backend",
                "runs",
                "avg wall ms",
                "std wall ms",
                "avg CPU ms",
                "CPU util.",
                "RTF",
            ],
            rows,
        )
    )


def save_json(
    path: Path,
    verification: list[VerificationResult],
    accuracy: list[AccuracyResult],
    benchmarks: list[BenchmarkResult],
) -> None:
    payload = {
        "verification": [asdict(item) for item in verification],
        "accuracy": [asdict(item) for item in accuracy],
        "benchmarks": [asdict(item) for item in benchmarks],
    }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON results saved to: {path}")


def main() -> None:
    args = parse_args()
    audio_paths = validate_args(args)

    # The course comparison labels this backend as PyTorch CPU. Force the same
    # behavior even if USE_CUDA was set in the caller's shell.
    os.environ["USE_CUDA"] = "0"

    names = selected_backend_names(args.backend)
    modules = {name: load_backend(name) for name in names}

    verification: list[VerificationResult] = []
    accuracy: list[AccuracyResult] = []
    if not args.skip_accuracy:
        verification, accuracy = verify_outputs(audio_paths, modules)
        print_verification(verification)
        if accuracy:
            print_accuracy(accuracy)
        elif args.backend != "all":
            print(
                "\nNumerical error was not calculated because Librosa was not "
                "selected as the reference backend."
            )

    benchmark_results = benchmark(audio_paths, modules, args.runs)
    print_benchmarks(benchmark_results)

    print("\nNotes:")
    print("- Audio loading/resampling is excluded from the timed region.")
    print("- One untimed warm-up is performed for every file/backend pair.")
    print("- CPU time is measured over the complete run block for higher precision.")
    print("- Use --backend with Memray to profile only one implementation.")

    if args.json_output is not None:
        save_json(
            args.json_output,
            verification=verification,
            accuracy=accuracy,
            benchmarks=benchmark_results,
        )


if __name__ == "__main__":
    main()
