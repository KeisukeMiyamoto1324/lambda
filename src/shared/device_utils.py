from pathlib import Path
import os
import time

import torch


DevicesValue = int | str
CUDA_DEVICE = torch.device("cuda")


def resolve_devices(devices: str) -> DevicesValue:
    # ---------------------------------------------------------
    # Convert the CLI device selector into the value expected by
    # Lightning while rejecting invalid numeric values early.
    # ---------------------------------------------------------
    if devices == "auto":
        return devices

    device_count = int(devices)

    if device_count < 1:
        raise ValueError("--devices must be auto or a positive integer")

    return device_count


def resolve_device_count(devices: DevicesValue) -> int:
    # ---------------------------------------------------------
    # Resolve the effective device count used for metadata and
    # validation budgets before Lightning builds the trainer.
    # ---------------------------------------------------------
    if isinstance(devices, int):
        return devices

    return torch.cuda.device_count()


def resolve_strategy(device_count: int) -> str | None:
    # ---------------------------------------------------------
    # Use explicit DDP when more than one CUDA device is active.
    # ---------------------------------------------------------
    if device_count > 1:
        return "ddp"

    return None


def is_global_zero_process() -> bool:
    # ---------------------------------------------------------
    # Check the distributed rank from the environment before the
    # Lightning trainer is available.
    # ---------------------------------------------------------
    return int(os.environ.get("RANK", "0")) == 0


def wait_for_file(path: Path, timeout_seconds: int = 3600) -> None:
    # ---------------------------------------------------------
    # Let non-zero ranks wait until rank zero finishes creating
    # shared files such as validation caches.
    # ---------------------------------------------------------
    start_time = time.monotonic()

    while not path.exists():
        if time.monotonic() - start_time > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for {path}")

        time.sleep(1)
