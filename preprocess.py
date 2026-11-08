from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm


WAVELENGTH = 1.55
FS = 1000
FFTLEN = 1024
F_RES = FS / FFTLEN
F_OFFSET = 79.29
F_CENTER_IDEAL = 82

FFT_OUT_START = 33
FFT_OUT_STOP = 132
FFT_OUT_LENGTH = FFT_OUT_STOP - FFT_OUT_START + 1
FFTSEGLEN_INVALID = 7
STEP_RECORDS = 360
SPLIT_ORDER = ("train", "val", "test")
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def parse_bin_file(file_path: Path) -> Tuple[np.ndarray, int]:
    data_original = np.fromfile(file_path, dtype=np.uint16)
    fftseglen = int(len(data_original) / FFT_OUT_LENGTH)
    total = fftseglen * FFT_OUT_LENGTH
    if total <= 0:
        raise ValueError(f"Empty or invalid binary payload: {file_path}")

    if data_original[0] == 0:
        data = np.empty(total, dtype=np.float32)
        data[: total - 1] = data_original[1:total]
        data[total - 1] = data[total - 2] if total > 1 else 0.0
    else:
        data = data_original[:total].astype(np.float32, copy=False)

    data_matrix = np.reshape(data, (FFT_OUT_LENGTH, fftseglen), order="F")
    ground_noise = np.mean(data_matrix[:, -3:], axis=1)
    data_matrix = data_matrix - ground_noise[:, None]
    sp_valid = np.abs(data_matrix[:, FFTSEGLEN_INVALID:fftseglen])
    return sp_valid, fftseglen


def get_distance_resolution(fftseglen: int) -> Tuple[int, int]:
    if fftseglen == 420:
        return 15, 21
    if fftseglen == 400:
        return 30, 11
    if fftseglen == 300:
        return 45, 7
    if fftseglen == 240:
        return 60, 5
    raise ValueError(f"Unsupported FFTSEGLEN: {fftseglen}")


def calculate_wind_speed(spectrum: np.ndarray, window_len: int) -> np.ndarray:
    n_freq, n_gate = spectrum.shape
    wind_speed = np.zeros(n_gate, dtype=np.float32)

    cnt = (window_len - 1) // 2
    f_center = FFT_OUT_LENGTH / 2.0 + (F_OFFSET / F_RES - F_CENTER_IDEAL)
    x = np.arange(1, n_freq + 1)

    for gate_idx in range(n_gate):
        y = spectrum[:, gate_idx]
        idx = int(np.argmax(y)) + 1
        idx_start = max(1, idx - cnt)
        idx_stop = min(n_freq, idx + cnt)

        mask = (idx_start <= x) & (x <= idx_stop)
        signal = y[mask]
        if signal.sum() > 0:
            weight = np.arange(idx_start, idx_stop + 1)
            fd = (np.sum(weight * signal) / np.sum(signal) - f_center) * F_RES
            wind_speed[gate_idx] = (WAVELENGTH / 2.0) * fd

    return wind_speed


def calculate_cnr(spectrum: np.ndarray, window_len: int) -> np.ndarray:
    n_freq, n_gate = spectrum.shape
    cnr = np.zeros(n_gate, dtype=np.float32)

    cnt = (window_len - 1) // 2
    x = np.arange(1, n_freq + 1)

    for gate_idx in range(n_gate):
        y = spectrum[:, gate_idx]
        idx = int(np.argmax(y)) + 1
        idx_start = max(1, idx - cnt)
        idx_stop = min(n_freq, idx + cnt)

        mask = (idx_start <= x) & (x <= idx_stop)
        signal = y[mask]
        noise = y[~mask]
        if len(signal) > 0 and len(noise) > 0:
            power_signal = np.sum(signal ** 2) / len(signal)
            power_noise = np.sum(noise ** 2) / len(noise)
            cnr[gate_idx] = 10 * np.log10(power_signal / (power_noise + 1e-12))

    return cnr


def calculate_snr(spectrum: np.ndarray) -> np.ndarray:
    n_freq, _ = spectrum.shape
    signal = np.max(spectrum, axis=0)
    power_signal = signal ** 2
    noise = spectrum[:, -1]
    power_noise = np.sum(noise ** 2) / n_freq
    return 10 * np.log10(power_signal / (power_noise + 1e-12))


def calculate_spectral_width(spectrum: np.ndarray) -> np.ndarray:
    _, n_gate = spectrum.shape
    fwhm = np.zeros(n_gate, dtype=np.float32)
    for gate_idx in range(n_gate):
        y = spectrum[:, gate_idx]
        half_max = np.max(y) / 2.0
        above_half = y > half_max
        if above_half.sum() > 0:
            indices = np.where(above_half)[0]
            fwhm[gate_idx] = indices[-1] - indices[0] + 1
    return fwhm


def calculate_spectral_moments(spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_freq, n_gate = spectrum.shape
    skewness = np.zeros(n_gate, dtype=np.float32)
    kurtosis = np.zeros(n_gate, dtype=np.float32)
    x = np.arange(n_freq, dtype=float)

    for gate_idx in range(n_gate):
        p = spectrum[:, gate_idx]
        p_sum = p.sum()
        if p_sum <= 0:
            continue

        mu = np.sum(x * p) / p_sum
        diff = x - mu
        m2 = np.sum(diff ** 2 * p) / p_sum
        m3 = np.sum(diff ** 3 * p) / p_sum
        m4 = np.sum(diff ** 4 * p) / p_sum

        if m2 > 0:
            skewness[gate_idx] = m3 / (m2 ** 1.5)
            kurtosis[gate_idx] = m4 / (m2 ** 2)

    return skewness, kurtosis


def process_bin_directory(
    input_dir: Path,
    max_files: Optional[int] = None,
    max_gates: int = 20,
) -> Dict[str, np.ndarray]:
    bin_files = sorted(input_dir.glob("*.bin"))
    if max_files is not None:
        bin_files = bin_files[:max_files]

    print(f"Processing {len(bin_files)} files...")

    n_files = len(bin_files)
    spectra = None
    wind_speed_all = None
    cnr_all = None
    snr_all = None
    sp_width_all = None
    skewness_all = None
    kurtosis_all = None
    target_gates: Optional[int] = None
    valid_count = 0

    for file_path in tqdm(bin_files):
        try:
            spectrum, fftseglen = parse_bin_file(file_path)
            _, window_len = get_distance_resolution(fftseglen)

            if target_gates is None:
                target_gates = max_gates if max_gates > 0 else spectrum.shape[1]
                spectra = np.zeros((n_files, target_gates, FFT_OUT_LENGTH), dtype=np.float32)
                wind_speed_all = np.zeros((n_files, target_gates), dtype=np.float32)
                cnr_all = np.zeros((n_files, target_gates), dtype=np.float32)
                snr_all = np.zeros((n_files, target_gates), dtype=np.float32)
                sp_width_all = np.zeros((n_files, target_gates), dtype=np.float32)
                skewness_all = np.zeros((n_files, target_gates), dtype=np.float32)
                kurtosis_all = np.zeros((n_files, target_gates), dtype=np.float32)

            assert target_gates is not None
            n_gate = min(target_gates, spectrum.shape[1])
            spectrum = spectrum[:, :n_gate].astype(np.float32, copy=False)

            wind_speed_single = calculate_wind_speed(spectrum, window_len)
            cnr_single = calculate_cnr(spectrum, window_len)
            snr_single = calculate_snr(spectrum)
            sp_width_single = calculate_spectral_width(spectrum)
            skew, kurt = calculate_spectral_moments(spectrum)

            spectra[valid_count, :n_gate, :] = spectrum.T
            wind_speed_all[valid_count, :n_gate] = wind_speed_single[:n_gate]
            cnr_all[valid_count, :n_gate] = cnr_single[:n_gate]
            snr_all[valid_count, :n_gate] = snr_single[:n_gate]
            sp_width_all[valid_count, :n_gate] = sp_width_single[:n_gate]
            skewness_all[valid_count, :n_gate] = skew[:n_gate]
            kurtosis_all[valid_count, :n_gate] = kurt[:n_gate]
            valid_count += 1
        except Exception as exc:
            print(f"Error processing {file_path}: {exc}")

    if valid_count == 0 or spectra is None:
        raise ValueError("No valid .bin files were processed.")

    return {
        "spectra": spectra[:valid_count],
        "wind_speed": wind_speed_all[:valid_count],
        "cnr": cnr_all[:valid_count],
        "snr": snr_all[:valid_count],
        "sp_width": sp_width_all[:valid_count],
        "skewness": skewness_all[:valid_count],
        "kurtosis": kurtosis_all[:valid_count],
    }


def load_existing_raw_npz(raw_npz: Path) -> Dict[str, np.ndarray]:
    data = np.load(raw_npz, allow_pickle=False)
    return {key: data[key] for key in data.files}


def regroup_records_to_steps(
    data: Dict[str, np.ndarray],
    step_records: int = STEP_RECORDS,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    total_records = int(data["spectra"].shape[0])
    usable_records = (total_records // step_records) * step_records
    dropped_records = total_records - usable_records
    num_steps = usable_records // step_records

    if num_steps <= 0:
        raise ValueError(
            f"Not enough records to form a full step: total_records={total_records}, "
            f"step_records={step_records}."
        )

    grouped: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.shape[0] != total_records:
            continue
        trimmed = value[:usable_records]
        reshaped = trimmed.reshape(num_steps, step_records, *value.shape[1:])
        if reshaped.shape[1] != step_records:
            raise AssertionError(
                f"Step regroup failed for {key}: expected {step_records}, got {reshaped.shape[1]}."
            )
        grouped[key] = reshaped.mean(axis=1).astype(np.float32)

    return grouped, {
        "total_records": total_records,
        "usable_records": usable_records,
        "dropped_records": dropped_records,
        "step_records": step_records,
        "num_steps": num_steps,
    }


def split_step_data(
    step_data: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, int]]:
    n_steps = int(step_data["spectra"].shape[0])
    train_end = int(n_steps * SPLIT_RATIOS["train"])
    val_end = train_end + int(n_steps * SPLIT_RATIOS["val"])

    if train_end <= 0 or val_end <= train_end or val_end >= n_steps:
        raise ValueError(
            f"Invalid 70/15/15 split for {n_steps} steps: "
            f"train_end={train_end}, val_end={val_end}."
        )

    boundaries = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_steps),
    }
    split_data: Dict[str, Dict[str, np.ndarray]] = {}
    split_log: Dict[str, int] = {}

    for split_name, (start, end) in boundaries.items():
        split_payload = {key: value[start:end] for key, value in step_data.items()}
        split_payload["step_indices"] = np.arange(start, end, dtype=np.int64)
        split_data[split_name] = split_payload
        split_log[f"{split_name}_start"] = start
        split_log[f"{split_name}_end"] = end
        split_log[f"{split_name}_steps"] = end - start

    if not (
        split_log["train_end"] == split_log["val_start"]
        and split_log["val_end"] == split_log["test_start"]
    ):
        raise AssertionError("Chronological split boundaries are not contiguous.")

    return split_data, split_log


def normalize_splits(
    split_data: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float]]:
    train_spectra = split_data["train"]["spectra"]
    train_wind = split_data["train"]["wind_speed"]
    spec_mean = float(train_spectra.mean())
    spec_std = float(train_spectra.std() + 1e-6)
    wind_mean = float(train_wind.mean())
    wind_std = float(train_wind.std() + 1e-6)

    stats = {
        "spec_mean": spec_mean,
        "spec_std": spec_std,
        "wind_mean": wind_mean,
        "wind_std": wind_std,
    }

    normalized: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name, payload in split_data.items():
        current: Dict[str, np.ndarray] = {}
        for key, value in payload.items():
            if key == "spectra":
                current[key] = ((value - spec_mean) / spec_std).astype(np.float32)
            elif key == "wind_speed":
                current[key] = ((value - wind_mean) / wind_std).astype(np.float32)
            else:
                current[key] = value
        normalized[split_name] = current
    return normalized, stats


def compute_window_count(
    num_steps: int,
    input_steps: int,
    output_steps: int,
    stride: int,
) -> int:
    if input_steps <= 0 or output_steps <= 0:
        raise ValueError("input_steps and output_steps must be positive integers.")
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")
    available = num_steps - input_steps - output_steps
    if available < 0:
        return 0
    return available // stride + 1


def validate_split_payload(
    split_name: str,
    payload: Dict[str, np.ndarray],
    input_steps: int,
    output_steps: int,
    stride: int,
) -> Dict[str, int]:
    spectra = payload["spectra"]
    wind_speed = payload["wind_speed"]
    if spectra.ndim != 3:
        raise AssertionError(f"{split_name} spectra must be [steps, gate, freq], got {spectra.shape}.")
    if wind_speed.ndim != 2:
        raise AssertionError(f"{split_name} wind_speed must be [steps, gate], got {wind_speed.shape}.")
    if spectra.shape[0] != wind_speed.shape[0]:
        raise AssertionError(
            f"{split_name} step mismatch: spectra={spectra.shape[0]}, wind={wind_speed.shape[0]}."
        )

    num_windows = compute_window_count(
        num_steps=int(spectra.shape[0]),
        input_steps=input_steps,
        output_steps=output_steps,
        stride=stride,
    )
    if num_windows <= 0:
        raise ValueError(
            f"{split_name} split cannot form any window with input_steps={input_steps}, "
            f"output_steps={output_steps}, stride={stride}, num_steps={spectra.shape[0]}."
        )
    return {
        "num_steps": int(spectra.shape[0]),
        "num_windows": num_windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess radar data into strict step-based temporal splits")
    parser.add_argument("--input-dir", type=str, default="data/raw_bin/NR-I", help="Directory with source .bin files")
    parser.add_argument("--raw-npz", type=str, default=None, help="Optional existing raw_data.npz to reuse")
    parser.add_argument("--output-dir", type=str, default="data/regenerated/NR-I", help="Output directory")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of .bin files to parse")
    parser.add_argument("--max-gates", type=int, default=60, help="Maximum number of gates to keep")
    parser.add_argument("--in-steps", type=int, default=12, help="Validation window history length")
    parser.add_argument("--out-steps", type=int, default=1, help="Validation window forecast horizon")
    parser.add_argument("--stride", type=int, default=1, help="Validation window stride")
    parser.add_argument("--step-records", type=int, default=STEP_RECORDS, help="Records per temporal step")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1: Load raw record-level data")
    print("=" * 60)
    if args.raw_npz:
        raw_data = load_existing_raw_npz(Path(args.raw_npz))
        print(f"Loaded existing raw npz: {args.raw_npz}")
    else:
        raw_data = process_bin_directory(
            input_dir=Path(args.input_dir),
            max_files=args.max_files,
            max_gates=args.max_gates,
        )
    print(f"record-level spectra shape: {raw_data['spectra'].shape}")
    print(f"record-level wind shape:   {raw_data['wind_speed'].shape}")

    raw_path = output_dir / "raw_data.npz"
    np.savez(raw_path, **raw_data)
    print(f"saved raw record-level data to: {raw_path}")

    print("\n" + "=" * 60)
    print("Step 2: Regroup records into strict steps")
    print("=" * 60)
    step_data, regroup_log = regroup_records_to_steps(raw_data, step_records=args.step_records)
    del raw_data
    print(json.dumps(regroup_log, indent=2, ensure_ascii=False))
    if regroup_log["dropped_records"] > 0:
        print(
            f"dropped trailing records: {regroup_log['dropped_records']} "
            f"(unable to form a full {args.step_records}-record step)"
        )

    print("\n" + "=" * 60)
    print("Step 3: Chronological 70/15/15 split on step sequence")
    print("=" * 60)
    split_data, split_log = split_step_data(step_data)
    print(json.dumps(split_log, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("Step 4: Train-only normalization on split sequences")
    print("=" * 60)
    normalized_splits, stats = normalize_splits(split_data)
    del split_data
    del step_data
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")

    validation_log: Dict[str, Dict[str, int]] = {}
    for split_name in SPLIT_ORDER:
        validation_log[split_name] = validate_split_payload(
            split_name=split_name,
            payload=normalized_splits[split_name],
            input_steps=args.in_steps,
            output_steps=args.out_steps,
            stride=args.stride,
        )
        print(
            f"{split_name}: steps={validation_log[split_name]['num_steps']}, "
            f"windows={validation_log[split_name]['num_windows']}"
        )

    train_path = output_dir / "train_sequence.npz"
    val_path = output_dir / "val_sequence.npz"
    test_path = output_dir / "test_sequence.npz"
    stats_path = output_dir / "normalization_stats.npz"
    params_path = output_dir / "params.json"
    log_path = output_dir / "preprocess_log.json"

    np.savez(train_path, split_name="train", step_records=args.step_records, **normalized_splits["train"])
    np.savez(val_path, split_name="val", step_records=args.step_records, **normalized_splits["val"])
    np.savez(test_path, split_name="test", step_records=args.step_records, **normalized_splits["test"])
    np.savez(stats_path, **stats)

    params = {
        "step_records": args.step_records,
        "input_steps": args.in_steps,
        "output_steps": args.out_steps,
        "stride": args.stride,
        "max_gates": args.max_gates,
        "n_freq": FFT_OUT_LENGTH,
        "regroup": regroup_log,
        "split": split_log,
        "window_validation": validation_log,
    }
    params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")

    log_payload = {
        "regroup": regroup_log,
        "split": split_log,
        "normalization_stats": stats,
        "window_validation": validation_log,
    }
    log_path.write_text(json.dumps(log_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"  train sequence: {train_path}")
    print(f"  val sequence:   {val_path}")
    print(f"  test sequence:  {test_path}")
    print(f"  stats:          {stats_path}")
    print(f"  params:         {params_path}")
    print(f"  log:            {log_path}")


if __name__ == "__main__":
    main()
