# action_normalizer_dp.py
# Exact Diffusion Policy-style action normalizer:
# - For abs-action-only: first 3 dims (pos) are min-max scaled to [-1, 1]
# - Remaining dims pass through (scale=1, offset=0), but info is filled like DP
# - Uses DP's exact small-range handling in get_range_normalizer_from_stat
#
# You can import:
#   from action_normalizer_dp import (
#       SingleFieldLinearNormalizer,
#       array_to_stats,
#       get_range_normalizer_from_stat,
#       robomimic_abs_action_only_normalizer_from_stat,
#   )

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

    # DP naming
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
    Exact DP -1..1 min-max with eps guard (same math as diffusion_policy):
      input_range = max - min
      ignore_dim = input_range < eps
      input_range[ignore_dim] = output_max - output_min  # makes scale=1.0 for those dims
      scale  = (output_max - output_min) / input_range
      offset = output_min - scale * input_min
      offset[ignore_dim] = (output_max + output_min)/2 - input_min   # = -input_min when out in [-1,1]
    """
    input_max = stat['max'].astype(np.float32)
    input_min = stat['min'].astype(np.float32)
    input_range = input_max - input_min

    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = (output_max - output_min)

    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2.0 - input_min[ignore_dim]  # -> -input_min when out range is [-1,1]

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

def robomimic_abs_action_only_normalizer_from_stat(stat: Dict[str, np.ndarray]) -> SingleFieldLinearNormalizer:
    """
    DP's 'abs-action-only' pattern:
      - Normalize pos (first 3 dims) to [-1,1] via min-max
      - Leave the rest identity (scale=1, offset=0), info filled as [-1,1], mean=0, std=1
    """
    # Split stats
    pos_stat = {k: stat[k][..., :3] for k in stat}
    other_stat = {k: stat[k][..., 3:] for k in stat}

    # Build pos normalizer
    pos_norm = get_range_normalizer_from_stat(pos_stat)

    # Identity for others
    example_other = other_stat['max'].astype(np.float32)
    scale_other = np.ones_like(example_other, dtype=np.float32)
    offset_other = np.zeros_like(example_other, dtype=np.float32)
    info_other = _identity_info_like(example_other)

    # Stitch scale/offset and info in DP order
    scale = np.concatenate([pos_norm.scale, scale_other], axis=-1)
    offset = np.concatenate([pos_norm.offset, offset_other], axis=-1)
    info = {
        'min': np.concatenate([pos_stat['min'], info_other['min']], axis=-1),
        'max': np.concatenate([pos_stat['max'], info_other['max']], axis=-1),
        'mean': np.concatenate([pos_stat['mean'], info_other['mean']], axis=-1),
        'std': np.concatenate([pos_stat['std'], info_other['std']], axis=-1),
    }

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=info
    )
