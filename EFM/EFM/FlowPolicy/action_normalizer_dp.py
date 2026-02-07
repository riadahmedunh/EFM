"""
action_normalizer_dp.py
Exact Diffusion Policy-style action normalizer.
Updated for User Constraint:
- Pose (XYZ + RPY, dims 0-6): Normalized to [-1, 1]
- Gripper (dim 6+): Identity (Raw)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class SingleFieldLinearNormalizer:
    """y = scale * x + offset (elementwise), mirroring DP's SingleFieldLinearNormalizer."""
    scale: np.ndarray
    offset: np.ndarray
    input_stats_dict: Dict[str, np.ndarray]

    @classmethod
    def create_manual(cls, scale: np.ndarray, offset: np.ndarray, input_stats_dict: Dict[str, np.ndarray]):
        return cls(
            scale=scale.astype(np.float32),
            offset=offset.astype(np.float32),
            input_stats_dict={k: v.astype(np.float32) for k, v in input_stats_dict.items()},
        )

    def encode(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.offset

    def decode(self, y: np.ndarray) -> np.ndarray:
        return (y - self.offset) / (self.scale + 1e-12)

    # Convenience aliases
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return self.encode(x)

    def denormalize(self, y: np.ndarray) -> np.ndarray:
        return self.decode(y)


# ---------- helpers matching DP logic ----------
def array_to_stats(arr: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        'min': arr.min(axis=0),
        'max': arr.max(axis=0),
        'mean': arr.mean(axis=0),
        'std': arr.std(axis=0)
    }

def get_range_normalizer_from_stat(stat: Dict[str, np.ndarray],
                                   output_max: float = 1.0,
                                   output_min: float = -1.0,
                                   range_eps: float = 1e-7) -> SingleFieldLinearNormalizer:
    """
    Exact DP -1..1 min-max with eps guard.
    """
    input_max = stat['max'].astype(np.float32)
    input_min = stat['min'].astype(np.float32)
    input_range = input_max - input_min

    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = (output_max - output_min)

    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2.0 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def _identity_info_like(example: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        'max': np.ones_like(example, dtype=np.float32),
        'min': -np.ones_like(example, dtype=np.float32),
        'mean': np.zeros_like(example, dtype=np.float32),
        'std': np.ones_like(example, dtype=np.float32)
    }

def pose_normalized_gripper_raw_normalizer_from_stat(stat: Dict[str, np.ndarray]) -> SingleFieldLinearNormalizer:
    """
    Specific Logic:
      - Dims 0-6 (XYZ + RPY): Normalize to [-1, 1]
      - Dims 6+ (Gripper): Identity / Raw
    """
    # Split stats at index 6
    # Assuming action dim is 7 (3 pos + 3 rot + 1 gripper)
    pose_stat = {k: stat[k][..., :6] for k in stat}
    gripper_stat = {k: stat[k][..., 6:] for k in stat}

    # 1. Normalize Pose (XYZ+RPY) to [-1, 1]
    pose_norm = get_range_normalizer_from_stat(pose_stat, output_min=-1.0, output_max=1.0)

    # 2. Keep Gripper Identity
    example_gripper = gripper_stat['max'].astype(np.float32)
    scale_gripper = np.ones_like(example_gripper, dtype=np.float32)
    offset_gripper = np.zeros_like(example_gripper, dtype=np.float32)
    info_gripper = _identity_info_like(example_gripper)

    # Stitch together
    scale = np.concatenate([pose_norm.scale, scale_gripper], axis=-1)
    offset = np.concatenate([pose_norm.offset, offset_gripper], axis=-1)
    info = {
        'min': np.concatenate([pose_stat['min'], info_gripper['min']], axis=-1),
        'max': np.concatenate([pose_stat['max'], info_gripper['max']], axis=-1),
        'mean': np.concatenate([pose_stat['mean'], info_gripper['mean']], axis=-1),
        'std': np.concatenate([pose_stat['std'], info_gripper['std']], axis=-1),
    }

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=info
    )