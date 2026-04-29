# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""NVLink topology detection via nvidia-smi.

Parses `nvidia-smi topo -m` output to discover NVLink pairs. No ctypes/NVML
dependency — works with any driver version that ships nvidia-smi.
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

_NV_RE = re.compile(r"NV(\d+)")


@dataclass(frozen=True)
class NvLinkPair:
    gpu_a: int
    gpu_b: int
    width: int  # bonded NVLink count (e.g. 4 for NV4)

    @property
    def devices(self) -> Tuple[torch.device, torch.device]:
        return torch.device("cuda", self.gpu_a), torch.device("cuda", self.gpu_b)


@dataclass
class GpuTopology:
    gpu_count: int
    nvlink_pairs: List[NvLinkPair] = field(default_factory=list)
    _peer_map: Dict[int, Set[int]] = field(default_factory=dict, repr=False)

    def has_nvlink(self) -> bool:
        return len(self.nvlink_pairs) > 0

    def nvlink_peers(self, gpu_idx: int) -> List[int]:
        return sorted(self._peer_map.get(gpu_idx, set()))

    def nvlink_device_pairs(self) -> List[Tuple[torch.device, torch.device]]:
        return [p.devices for p in self.nvlink_pairs]

    def nvlink_connected_devices(self) -> List[torch.device]:
        seen: Set[int] = set()
        for p in self.nvlink_pairs:
            seen.add(p.gpu_a)
            seen.add(p.gpu_b)
        return [torch.device("cuda", i) for i in sorted(seen)]


def detect_topology() -> GpuTopology:
    """Detect GPU topology by parsing nvidia-smi topo -m."""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count < 2:
        return GpuTopology(gpu_count=gpu_count)

    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.debug("nvidia-smi topo failed: %s", result.stderr.strip())
            return GpuTopology(gpu_count=gpu_count)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("nvidia-smi topo unavailable: %s", exc)
        return GpuTopology(gpu_count=gpu_count)

    return _parse_topo_output(result.stdout, gpu_count)


def _parse_topo_output(output: str, gpu_count: int) -> GpuTopology:
    pairs: List[NvLinkPair] = []
    peer_map: Dict[int, Set[int]] = {}
    seen_pairs: Set[FrozenSet[int]] = set()

    lines = output.strip().splitlines()
    gpu_rows = [l for l in lines if l.startswith("GPU")]
    # Header row tells us column order
    header = next((l for l in lines if "\tGPU0" in l or "GPU0" in l), None)
    if header is None:
        return GpuTopology(gpu_count=gpu_count)

    header_cols = re.split(r"\s+", header.strip())
    gpu_col_indices = {}
    for ci, col in enumerate(header_cols):
        m = re.match(r"GPU(\d+)", col)
        if m:
            gpu_col_indices[ci] = int(m.group(1))

    for row in gpu_rows:
        parts = re.split(r"\t+", row.strip())
        if not parts:
            continue
        row_match = re.match(r"GPU(\d+)", parts[0])
        if not row_match:
            continue
        row_gpu = int(row_match.group(1))

        for ci, cell in enumerate(parts[1:], start=1):
            nv_match = _NV_RE.match(cell.strip())
            if not nv_match:
                continue
            col_gpu = gpu_col_indices.get(ci)
            if col_gpu is None:
                continue
            pair_key = frozenset((row_gpu, col_gpu))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            width = int(nv_match.group(1))
            a, b = min(row_gpu, col_gpu), max(row_gpu, col_gpu)
            pairs.append(NvLinkPair(gpu_a=a, gpu_b=b, width=width))
            peer_map.setdefault(a, set()).add(b)
            peer_map.setdefault(b, set()).add(a)

    if pairs:
        logger.info(
            "NVLink topology: %d pairs detected: %s",
            len(pairs),
            ", ".join(f"GPU{p.gpu_a}<->GPU{p.gpu_b} (NV{p.width})" for p in pairs),
        )

    return GpuTopology(gpu_count=gpu_count, nvlink_pairs=pairs, _peer_map=peer_map)


_cached_topology: Optional[GpuTopology] = None


def get_topology() -> GpuTopology:
    global _cached_topology
    if _cached_topology is None:
        _cached_topology = detect_topology()
    return _cached_topology
