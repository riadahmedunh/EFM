#!/usr/bin/env python3
"""
Hybrid Action Normalizer for GraspSplats
========================================

Combines the working robot movement normalization from the previous code
with the enhanced gripper normalization to get the best of both worlds.

Action vector format: [dx, dy, dz, rx, ry, rz, gripper_action]
- dx, dy, dz: translation deltas (normalize to [-1, 1]) - SAME AS WORKING VERSION
- rx, ry, rz: rotation deltas (identity normalization) - SAME AS WORKING VERSION
- gripper_action: gripper command (normalize to [0, 1] for open/close) - ENHANCED FIX
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

def hybrid_action_normalizer_from_stat(stat: Dict[str, np.ndarray]) -> SingleFieldLinearNormalizer:
    """
    HYBRID normalizer: Working movement + Enhanced gripper
    
    - Translation (xyz, dims 0-2): normalize to [-1,1] via min-max (SAME AS WORKING VERSION)
    - Rotation (rxryrz, dims 3-5): identity (scale=1, offset=0) (SAME AS WORKING VERSION)  
    - Gripper (dim 6): normalize to [0,1] via min-max for better control (ENHANCED FIX)
    """
    print("🔧 Using HYBRID GraspSplats action normalizer:")
    print("   • Translation (xyz): normalized to [-1, 1] (same as working version)")
    print("   • Rotation (rxryrz): identity normalization (same as working version)") 
    print("   • Gripper: normalized to [0, 1] for better control (ENHANCED FIX)")
    
    # Split stats: [dx,dy,dz], [rx,ry,rz], [gripper]
    pos_stat = {k: stat[k][..., :3] for k in stat}        # translation
    rot_stat = {k: stat[k][..., 3:6] for k in stat}       # rotation  
    gripper_stat = {k: stat[k][..., 6:7] for k in stat}   # gripper
    
    # 1. Normalize position to [-1, 1] (SAME AS WORKING VERSION)
    pos_norm = get_range_normalizer_from_stat(pos_stat, output_min=-1.0, output_max=1.0)
    
    # 2. Identity for rotation (rxryrz) (SAME AS WORKING VERSION)
    example_rot = rot_stat['max'].astype(np.float32)
    scale_rot = np.ones_like(example_rot, dtype=np.float32)
    offset_rot = np.zeros_like(example_rot, dtype=np.float32)
    info_rot = _identity_info_like(example_rot)
    
    # 3. Normalize gripper to [0, 1] for better control (ENHANCED FIX)
    gripper_norm = get_range_normalizer_from_stat(gripper_stat, output_min=0.0, output_max=1.0)
    
    # Print gripper normalization info
    gripper_min = gripper_stat['min'][0]
    gripper_max = gripper_stat['max'][0] 
    print(f"   • Gripper raw range: [{gripper_min:.4f}, {gripper_max:.4f}] -> [0.0, 1.0]")
    
    # Combine all normalizers
    scale = np.concatenate([pos_norm.scale, scale_rot, gripper_norm.scale], axis=-1)
    offset = np.concatenate([pos_norm.offset, offset_rot, gripper_norm.offset], axis=-1)
    
    # Combine stats info
    info = {
        'min': np.concatenate([pos_stat['min'], info_rot['min'], gripper_stat['min']], axis=-1),
        'max': np.concatenate([pos_stat['max'], info_rot['max'], gripper_stat['max']], axis=-1),
        'mean': np.concatenate([pos_stat['mean'], info_rot['mean'], gripper_stat['mean']], axis=-1),
        'std': np.concatenate([pos_stat['std'], info_rot['std'], gripper_stat['std']], axis=-1),
    }
    
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=info
    )
