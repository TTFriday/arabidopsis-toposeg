"""Measure parameters, MACs and CPU inference latency for every training configuration.

Writes outputs/cost/model_cost.csv (one row per configuration) and
outputs/cost/hardware.json (host details for the reproducibility note).
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import io
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from toposeg.inference import sliding_window_inference
from toposeg.models import CompactUNet
from toposeg.utils import load_config

CONFIGS = {
    "full": "configs/full.yaml",
    "baseline_region": "configs/baseline_region.yaml",
    "baseline_cldice": "configs/baseline_cldice.yaml",
    "baseline_skelrecall": "configs/baseline_skelrecall.yaml",
    "ablation_uniform": "configs/ablation_uniform.yaml",
    "ablation_no_joint_root": "configs/ablation_no_joint_root.yaml",
    "ablation_no_skeleton_head": "configs/ablation_no_skeleton_head.yaml",
}
SEEDS = (17, 29, 43)


def cpu_name() -> str:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name.Trim()"],
            capture_output=True, text=True, timeout=30, check=True)
        return out.stdout.strip()
    except Exception:
        return platform.processor() or "unknown"


def memory_bytes() -> tuple[int, int]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.wintypes.DWORD),
            ("dwMemoryLoad", ctypes.wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def parameter_count(module: nn.Module) -> int:
    return int(sum(int(p.numel()) for p in module.parameters()))


def state_dict_parameters(path: Path) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    return int(sum(int(v.numel()) for v in state.values() if hasattr(v, "numel")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--image", type=str, default="2464,3280", help="H,W of the full frame")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--reps", type=int, default=9, help="timed patch forward passes")
    parser.add_argument("--image-reps", type=int, default=3, help="timed full-image passes")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/cost"))
    parser.add_argument("--runs", type=Path, default=Path("outputs"),
                        help="root of the training runs; best.pt under it cross-checks the parameter count")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    total_phys, avail_phys = memory_bytes()
    hardware = {
        "device": device.type,
        "cpu_name": cpu_name(),
        "logical_processors": os_cpu_count(),
        "torch_threads": args.threads,
        "ram_total_gb": round(total_phys / 2**30, 1),
        "ram_available_gb_at_start": round(avail_phys / 2**30, 1),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "patch_size": args.patch,
        "image_h_w": args.image,
        "overlap": 0.5,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        hardware.update({
            "gpu_name": properties.name,
            "gpu_memory_gb": round(properties.total_memory / 2**30, 2),
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_peak_memory_allocated_gb_at_end": None,
        })
    (out_dir / f"hardware_{device.type}.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")

    height, width = (int(v) for v in args.image.split(","))
    rows = []
    for name, config_path in CONFIGS.items():
        config = load_config(config_path)
        model_config = config["model"]
        model = CompactUNet(
            base_channels=int(model_config.get("base_channels", 24)),
            skeleton_head=bool(model_config.get("skeleton_head", True)),
        ).to(device)
        model.eval()

        params = parameter_count(model)
        ckpt_params = []
        for seed in SEEDS:
            path = args.runs / name / f"seed_{seed}" / "best.pt"
            if path.is_file():
                ckpt_params.append(state_dict_parameters(path))
        if ckpt_params and len(set(ckpt_params + [params])) != 1:
            raise SystemExit(f"{name}: parameter counts disagree {params} vs {ckpt_params}")
        if len(ckpt_params) < len(SEEDS):
            print(f"note: {name}: cross-checked against {len(ckpt_params)} of {len(SEEDS)} checkpoints", flush=True)

        macs = measure_macs(model, args.patch)
        weights_bytes = dump_size(model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        latents = []
        with torch.no_grad():
            dummy = torch.randn(1, 1, args.patch, args.patch, device=device)
            for _ in range(3):
                model(dummy)
            sync(device)
            for _ in range(args.reps):
                start = time.perf_counter()
                model(dummy)
                sync(device)
                latents.append((time.perf_counter() - start) * 1000.0)

        row = {
            "configuration": name,
            "device": device.type,
            "architecture": "dual-head" if model_config.get("skeleton_head", True) else "region-only head",
            "parameters": params,
            "macs_per_patch_512": macs,
            "weights_file_mb": round(weights_bytes / 2**20, 1),
            f"{device.type}_ms_per_patch_median": round(statistics.median(latents), 1),
            f"{device.type}_ms_per_patch_min": round(min(latents), 1),
            f"{device.type}_ms_per_patch_max": round(max(latents), 1),
            "patch_reps": int(args.reps),
            "warmup_reps": 3,
        }

        if not args.skip_image:
            image = np.random.default_rng(0).random((height, width)).astype(np.float32)
            full_secs = []
            for _ in range(args.image_reps):
                sync(device)
                start = time.perf_counter()
                sliding_window_inference(model, image, patch_size=args.patch, overlap=0.5, device=device)
                sync(device)
                full_secs.append(time.perf_counter() - start)
            patches_per_image = _patch_count(height, width, args.patch, args.patch // 2)
            row.update({
                "patches_per_image": patches_per_image,
                "frame_reps": int(args.image_reps),
                f"{device.type}_s_per_image_median": round(statistics.median(full_secs), 2),
                f"{device.type}_s_per_image_min": round(min(full_secs), 2),
                f"{device.type}_s_per_image_max": round(max(full_secs), 2),
            })
            if device.type == "cuda":
                row["gpu_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            del image

        rows.append(row)
        print(json.dumps(row), flush=True)

    write_csv(rows, out_dir / f"model_cost_{device.type}.csv")
    print(f"wrote {out_dir / ('model_cost_' + device.type + '.csv')}")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def os_cpu_count() -> int:
    import os
    return os.cpu_count() or 0


def measure_macs(model: nn.Module, patch: int) -> int:
    total = [0]

    def conv_hook(module, inp, out):
        kernel = int(module.kernel_size[0]) * int(module.kernel_size[1])
        total[0] += (int(out.shape[-2]) * int(out.shape[-1]) * int(module.out_channels)
                     * kernel * (int(module.in_channels) // int(module.groups)))

    def convt_hook(module, inp, out):
        kernel = int(module.kernel_size[0]) * int(module.kernel_size[1])
        total[0] += (int(inp[0].shape[-2]) * int(inp[0].shape[-1]) * int(module.in_channels)
                     * kernel * (int(module.out_channels) // int(module.groups)))

    handles = []
    for module in model.modules():
        if isinstance(module, nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(convt_hook))
        elif isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, patch, patch, device=next(model.parameters()).device))
    for handle in handles:
        handle.remove()
    return int(total[0])


def dump_size(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save({k: v for k, v in model.state_dict().items()}, buffer)
    return buffer.tell()


def _patch_count(height: int, width: int, patch: int, stride: int) -> int:
    def starts(length: int) -> int:
        if length <= patch:
            return 1
        values = list(range(0, length - patch + 1, stride))
        if values[-1] != length - patch:
            values.append(length - patch)
        return len(values)

    return starts(height) * starts(width)


def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
