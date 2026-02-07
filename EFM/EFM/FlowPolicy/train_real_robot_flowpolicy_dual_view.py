#!/usr/bin/env python3
"""
Real Robot FlowPolicy Training with Dual View Point Clouds (AgentView + Eye-in-Hand).
Follows exact pattern from train_flowpolicy_exact_normalized_actions_v2.py
"""

import sys
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import zarr
from torch.utils.data import Dataset
from omegaconf import OmegaConf

# Add FlowPolicy to path
FLOWPOLICY_ROOT = pathlib.Path(__file__).parent / 'FlowPolicy' / 'FlowPolicy'
print(FLOWPOLICY_ROOT)
if FLOWPOLICY_ROOT.exists():
    sys.path.insert(0, str(FLOWPOLICY_ROOT))
    os.chdir(str(FLOWPOLICY_ROOT))

from flow_policy_3d.policy.flowpolicy import FlowPolicy
from flow_policy_3d.model.common.normalizer import LinearNormalizer
from flow_policy_3d.dataset.base_dataset import BaseDataset
sys.path.append("FlowPolicy")
from train import TrainFlowPolicyWorkspace

# Import DP action normalizer
from action_normalizer import array_to_stats, robomimic_abs_action_only_normalizer_from_stat, SingleFieldLinearNormalizer

# Import PointNet encoders
from flow_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZRGB, PointNetEncoderXYZ, create_mlp


class DualViewFlowPolicyEncoder(nn.Module):
    """
    Dual-view point cloud encoder for FlowPolicy.
    Processes agentview and eye_in_hand point clouds with separate PointNets,
    then concatenates their features with state features.
    """
    def __init__(self, 
                 observation_space: dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), 
                 state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='mlp',
                 ):
        super().__init__()
        self.state_key = 'agent_pos'
        self.agentview_key = 'point_cloud'  # agentview point cloud
        self.eye_in_hand_key = 'eye_in_hand_point_cloud'  # eye-in-hand point cloud
        
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        # Get shapes from observation space
        self.agentview_shape = observation_space[self.agentview_key]
        self.eye_in_hand_shape = observation_space[self.eye_in_hand_key]
        self.state_shape = observation_space[self.state_key]
        
        print(f"\n🔧 [DualViewFlowPolicyEncoder] Initializing dual-view encoder:")
        print(f"   • AgentView point cloud shape: {self.agentview_shape}")
        print(f"   • Eye-in-Hand point cloud shape: {self.eye_in_hand_shape}")
        print(f"   • State shape: {self.state_shape}")
        print(f"   • Use PC color: {use_pc_color}")
        print(f"   • PointNet type: {pointnet_type}")
        
        # Create two separate PointNet encoders (one for each view)
        if pointnet_type == "mlp":
            if use_pc_color:
                encoder_cfg_agentview = dict(pointcloud_encoder_cfg)
                encoder_cfg_agentview['in_channels'] = 6
                encoder_cfg_eye_in_hand = dict(pointcloud_encoder_cfg)
                encoder_cfg_eye_in_hand['in_channels'] = 6
                
                self.agentview_encoder = PointNetEncoderXYZRGB(**encoder_cfg_agentview)
                self.eye_in_hand_encoder = PointNetEncoderXYZRGB(**encoder_cfg_eye_in_hand)
                print(f"   • Created 2x PointNetEncoderXYZRGB (6 channels each)")
            else:
                encoder_cfg_agentview = dict(pointcloud_encoder_cfg)
                encoder_cfg_agentview['in_channels'] = 3
                encoder_cfg_eye_in_hand = dict(pointcloud_encoder_cfg)
                encoder_cfg_eye_in_hand['in_channels'] = 3
                
                self.agentview_encoder = PointNetEncoderXYZ(**encoder_cfg_agentview)
                self.eye_in_hand_encoder = PointNetEncoderXYZ(**encoder_cfg_eye_in_hand)
                print(f"   • Created 2x PointNetEncoderXYZ (3 channels each)")
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")
        
        # State MLP (same as original)
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = list(state_mlp_size[:-1])
        output_dim = state_mlp_size[-1]
        
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))
        
        # Total output: agentview_features + eye_in_hand_features + state_features
        # Each PointNet outputs out_channel, state MLP outputs output_dim
        self.n_output_channels = out_channel * 2 + output_dim  # 2x point cloud encoders + state
        
        print(f"   • Output dimension: {self.n_output_channels} (2x{out_channel} + {output_dim})")
        print(f"✅ [DualViewFlowPolicyEncoder] Initialized successfully!\n")

    def forward(self, observations: dict) -> torch.Tensor:
        # Get point clouds from both views
        agentview_points = observations[self.agentview_key]
        eye_in_hand_points = observations[self.eye_in_hand_key]
        state = observations[self.state_key]
        
        # Encode each point cloud separately
        agentview_feat = self.agentview_encoder(agentview_points)      # B x out_channel
        eye_in_hand_feat = self.eye_in_hand_encoder(eye_in_hand_points)  # B x out_channel
        
        # Encode state
        state_feat = self.state_mlp(state)  # B x 64
        
        # Concatenate all features
        final_feat = torch.cat([agentview_feat, eye_in_hand_feat, state_feat], dim=-1)
        
        return final_feat

    def output_shape(self):
        return self.n_output_channels


def custom_robot_action_normalizer_from_stat(stat):
    """
    Identity normalizer for robot actions - NO normalization applied.
    All action dimensions are kept in their original range.
    """
    import numpy as np
    
    # Create identity normalizer (scale=1, offset=0 for all dimensions)
    example_action = stat['max'].astype(np.float32)
    scale = np.ones_like(example_action, dtype=np.float32)
    offset = np.zeros_like(example_action, dtype=np.float32)
    
    # Create identity info
    from action_normalizer import _identity_info_like
    info = _identity_info_like(example_action)
    
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=info
    )


class FlowPolicyWithActionMSE(FlowPolicy):
    """
    Enhanced FlowPolicy with CFM Theory Fixes - EXACT COPY from working code
    
    FIXES IMPLEMENTED:
    1. Better Integration: RK4 instead of Euler for accuracy
    2. Multi-Step CFM Training: Full trajectory consistency 
    3. Velocity Regularization: Global consistency terms
    """
    
    def __init__(self, use_action_mse=True, **kwargs):
        super().__init__(**kwargs)
        self.use_action_mse = use_action_mse
        print("🎯 Enhanced FlowPolicy with CFM Theory Fixes:")
        print(f"   • RK4 Integration for better accuracy")
        print(f"   • Multi-step CFM training for trajectory consistency") 
        print(f"   • Velocity regularization for global consistency")
        print(f"   • CFM delta: {self.delta}")
        print(f"   • CFM eps: {self.eps}")
        print(f"   • Use action MSE: {use_action_mse}")
        
        # Initialize epoch-level loss tracking
        self.epoch_losses = {
            'cfm_loss': [],
            'multi_step_loss': [],
            'velocity_reg_loss': [],
            'total_loss': [],
            'action_mse': []
        }
        self.current_epoch = 0
        self.batch_count = 0
        
        # Initialize logging
        self.log_dir = None
        self.epoch_log_file = None
        self.batch_log_file = None
    
    def setup_logging(self, log_dir):
        """Setup logging directory and files."""
        import os
        from datetime import datetime
        
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create epoch summary log
        self.epoch_log_file = self.log_dir / f"epoch_losses_{timestamp}.csv"
        with open(self.epoch_log_file, 'w') as f:
            f.write("epoch,cfm_loss_avg,cfm_loss_std,multi_step_loss_avg,multi_step_loss_std,")
            f.write("velocity_reg_loss_avg,velocity_reg_loss_std,total_loss_avg,total_loss_std,")
            f.write("action_mse_avg,action_mse_std,num_batches\n")
        
        # Create per-batch log
        self.batch_log_file = self.log_dir / f"batch_losses_{timestamp}.csv"
        with open(self.batch_log_file, 'w') as f:
            f.write("epoch,batch,cfm_loss,multi_step_loss,velocity_reg_loss,total_loss,action_mse\n")
        
        print(f"📝 Logging initialized:")
        print(f"   • Epoch log: {self.epoch_log_file}")
        print(f"   • Batch log: {self.batch_log_file}")
    
    def rk4_integration_step(self, z, t, dt, local_cond, global_cond, cond_data, cond_mask):
        """
        FIX 1: RK4 Integration - Higher order accuracy than Euler
        
        RK4 provides much better integration accuracy for CFM velocity field
        """
        Da = self.action_dim
        
        def get_velocity(z_curr, t_curr):
            pred = self.model(z_curr, t_curr*99, local_cond=local_cond, global_cond=global_cond)
            pred[cond_mask] = cond_data[cond_mask]
            return pred
        
        # RK4 coefficients
        k1 = get_velocity(z, t) * dt
        k2 = get_velocity(z + k1/2, t + dt/2) * dt  
        k3 = get_velocity(z + k2/2, t + dt/2) * dt
        k4 = get_velocity(z + k3, t + dt) * dt
        
        # RK4 update
        z_new = z + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Apply conditioning
        z_new[cond_mask] = cond_data[cond_mask]
        
        return z_new
    
    def predict_action(self, obs_dict):
        """
        ENHANCED prediction with RK4 integration for better accuracy
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            if 'eye_in_hand_point_cloud' in nobs:
                nobs['eye_in_hand_point_cloud'] = nobs['eye_in_hand_point_cloud'][..., :3]
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        # handle conditioning (same as original)
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(B, -1)
            
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            local_cond = None
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            Do = nobs_features.shape[-1]
            
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
            global_cond = None
        
        # Initialize from noise
        noise = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=cond_data.device)
        z = noise.detach().clone()
        
        # Enhanced integration parameters
        eps = self.eps
        delta = self.delta  
        
        # Calculate steps for better accuracy
        num_steps = max(10, int((1 - eps) / delta))  # More steps for RK4
        dt = (1 - eps) / num_steps
        
        # RK4 Integration loop
        for i in range(num_steps):
            t_val = eps + i * dt
            t = torch.ones(z.shape[0], device=device) * t_val
            
            # Use RK4 integration step
            z = self.rk4_integration_step(z, t, dt, local_cond, global_cond, cond_data, cond_mask)
        
        # Extract actions
        naction_pred = z[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        return {
            'action': action,
            'action_pred': action_pred,
        }
    
    def compute_loss(self, batch):
        """
        ENHANCED LOSS: CFM + Multi-Step Training + Velocity Regularization
        
        FIX 2: Multi-Step CFM Training for trajectory consistency
        FIX 3: Velocity Regularization for global consistency
        """
        # Standard CFM loss
        cfm_loss, loss_dict = super().compute_loss(batch)
        
        # Enhanced loss components
        multi_step_loss = self.compute_multi_step_cfm_loss(batch)
        velocity_reg_loss = self.compute_velocity_regularization_loss(batch)
        
        # Weights for different loss components
        cfm_weight = 1.0
        multi_step_weight = 0.5  # FIX 2: Multi-step consistency
        velocity_reg_weight = 0.5  # FIX 3: Velocity smoothness
        action_mse_weight = 0.1 if self.use_action_mse else 0.0
        
        # Total enhanced loss
        total_loss = (cfm_weight * cfm_loss + 
                     multi_step_weight * multi_step_loss +
                     velocity_reg_weight * velocity_reg_loss)
        
        # Optional action MSE supervision
        if self.use_action_mse:
            try:
                # Get the normalized observations and actions
                nobs = self.normalizer.normalize(batch['obs'])
                nactions = self.normalizer['action'].normalize(batch['action'])
                target_actions = nactions  # Target actions
                
                if not self.use_pc_color:
                    nobs['point_cloud'] = nobs['point_cloud'][..., :3]
                    if 'eye_in_hand_point_cloud' in nobs:
                        nobs['eye_in_hand_point_cloud'] = nobs['eye_in_hand_point_cloud'][..., :3]
                
                # Get observation encoding (same as in CFM loss)
                batch_size = nactions.shape[0]
                if self.obs_as_global_cond:
                    this_nobs = dict_apply(nobs, 
                        lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                    nobs_features = self.obs_encoder(this_nobs)
                    global_cond = nobs_features.reshape(batch_size, -1)
                    local_cond = None
                else:
                    this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
                    nobs_features = self.obs_encoder(this_nobs)
                    global_cond = None
                    local_cond = nobs_features  # Use as local conditioning
                
                # Simplified action prediction (for loss computation with gradients)
                device = nactions.device
                dtype = nactions.dtype
                
                # Initialize from noise
                B, T, Da = nactions.shape
                noise = torch.randn(B, T, Da, device=device, dtype=dtype)
                
                # Single-step prediction (simplified for loss computation)
                t_final = torch.ones(B, device=device) * 0.9  # Near final time
                predicted_velocity = self.model(noise, t_final*99, local_cond=local_cond, global_cond=global_cond)
                
                # Simple integration: x_final = noise + velocity * dt
                dt = 0.9  # From noise (t=0) to final (t=0.9)
                predicted_actions = noise + predicted_velocity * dt
                
                # Compute direct MSE between predicted and target actions
                action_mse = torch.mean((predicted_actions - target_actions) ** 2)
                total_loss = total_loss + action_mse_weight * action_mse
                
                loss_dict['action_mse'] = action_mse.item()
            except Exception as e:
                loss_dict['action_mse'] = float('nan')
                print(f"⚠️ Action MSE computation failed: {e}")
        
        # Update loss dictionary
        loss_dict.update({
            'cfm_loss_original': cfm_loss.item(),
            'multi_step_loss': multi_step_loss.item(),
            'velocity_reg_loss': velocity_reg_loss.item(),
            'total_enhanced_loss': total_loss.item()
        })
        
        # Track losses for epoch summary
        self.epoch_losses['cfm_loss'].append(cfm_loss.item())
        self.epoch_losses['multi_step_loss'].append(multi_step_loss.item())
        self.epoch_losses['velocity_reg_loss'].append(velocity_reg_loss.item())
        self.epoch_losses['total_loss'].append(total_loss.item())
        if self.use_action_mse and 'action_mse' in loss_dict:
            self.epoch_losses['action_mse'].append(loss_dict['action_mse'])
        
        # Print detailed loss components for every batch
        action_mse_val = loss_dict.get('action_mse', 0.0)
        print(f"[Epoch {self.current_epoch} | Batch {self.batch_count:4d}] "
              f"CFM: {cfm_loss.item():.6f} | "
              f"MultiStep: {multi_step_loss.item():.6f} | "
              f"VelReg: {velocity_reg_loss.item():.6f} | "
              f"ActionMSE: {action_mse_val:.6f} | "
              f"TOTAL: {total_loss.item():.6f}")
        
        # Log to batch CSV file
        if self.batch_log_file:
            with open(self.batch_log_file, 'a') as f:
                f.write(f"{self.current_epoch},{self.batch_count},")
                f.write(f"{cfm_loss.item():.8f},{multi_step_loss.item():.8f},")
                f.write(f"{velocity_reg_loss.item():.8f},{total_loss.item():.8f},")
                f.write(f"{action_mse_val:.8f}\n")
        
        self.batch_count += 1
        
        return total_loss, loss_dict
    
    def print_epoch_summary(self):
        """Print summary of all loss components for the completed epoch."""
        if not self.epoch_losses['cfm_loss']:
            return
            
        import numpy as np
        
        print("\n" + "="*80)
        print(f"📊 EPOCH {self.current_epoch} SUMMARY - Loss Components")
        print("="*80)
        
        stats = {}
        for loss_name, loss_values in self.epoch_losses.items():
            if loss_values:
                avg = np.mean(loss_values)
                std = np.std(loss_values)
                min_val = np.min(loss_values)
                max_val = np.max(loss_values)
                
                stats[loss_name] = {'avg': avg, 'std': std, 'min': min_val, 'max': max_val}
                
                print(f"{loss_name:20s}: avg={avg:8.6f}  std={std:8.6f}  min={min_val:8.6f}  max={max_val:8.6f}")
        
        num_batches = len(self.epoch_losses['cfm_loss'])
        print(f"\nTotal batches in epoch: {num_batches}")
        print("="*80 + "\n")
        
        # Write to epoch log CSV
        if self.epoch_log_file:
            with open(self.epoch_log_file, 'a') as f:
                f.write(f"{self.current_epoch},")
                f.write(f"{stats['cfm_loss']['avg']:.8f},{stats['cfm_loss']['std']:.8f},")
                f.write(f"{stats['multi_step_loss']['avg']:.8f},{stats['multi_step_loss']['std']:.8f},")
                f.write(f"{stats['velocity_reg_loss']['avg']:.8f},{stats['velocity_reg_loss']['std']:.8f},")
                f.write(f"{stats['total_loss']['avg']:.8f},{stats['total_loss']['std']:.8f},")
                
                if 'action_mse' in stats:
                    f.write(f"{stats['action_mse']['avg']:.8f},{stats['action_mse']['std']:.8f},")
                else:
                    f.write("0.0,0.0,")
                
                f.write(f"{num_batches}\n")
        
        # Reset for next epoch
        for key in self.epoch_losses:
            self.epoch_losses[key] = []
        self.current_epoch += 1
    
    def compute_multi_step_cfm_loss(self, batch):
        """
        FIX 2: Multi-Step CFM Training
        
        Train on full trajectory consistency instead of single-step velocity prediction.
        This ensures the velocity field produces globally consistent paths.
        """
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])
        
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            if 'eye_in_hand_point_cloud' in nobs:
                nobs['eye_in_hand_point_cloud'] = nobs['eye_in_hand_point_cloud'][..., :3]
        
        B, T, Da = naction.shape  # Get tensor dimensions
        To = nobs['point_cloud'].shape[1]  # obs timesteps
        device = naction.device
        
        # Setup conditioning (same as main compute_loss)
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(B, -1)
            local_cond = None
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            global_cond = None
            local_cond = nobs_features

        seg_losses = []
        num_segments = 2  # Reduced for faster, more consistent compute
        steps_per_seg = 16  # Reduced for faster compute
        for _ in range(num_segments):
            t0 = torch.rand(B, 1, 1, device=device) * 0.7         # [0, 0.7]
            t1 = t0 + torch.rand(B, 1, 1, device=device) * 0.3    # [t0, 1.0]

            z = torch.randn_like(naction)
            x0 = (1 - t0) * z + t0 * naction   # γ(t0)
            x1 = (1 - t1) * z + t1 * naction   # γ(t1)
            target_delta = x1 - x0

            x = x0.clone()
            # integrate v(x,t) from t0→t1
            for s in range(steps_per_seg):
                ts = (t0 + (s / steps_per_seg) * (t1 - t0)).squeeze(-1).squeeze(-1)  # (B,)
                v = self.model(x, ts * 99.0, local_cond=local_cond, global_cond=global_cond)
                dt = (t1 - t0) / steps_per_seg
                x = x + v * dt

            pred_delta = x - x0
            seg_losses.append(torch.nn.functional.mse_loss(pred_delta, target_delta))

        return torch.stack(seg_losses).mean()
    
    def compute_velocity_regularization_loss(self, batch):
        """
        FIX 3: Velocity Regularization
        
        Add smoothness and consistency constraints to velocity field.
        This ensures global consistency of the learned flow.
        """
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])
        
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            if 'eye_in_hand_point_cloud' in nobs:
                nobs['eye_in_hand_point_cloud'] = nobs['eye_in_hand_point_cloud'][..., :3]
        
        B, T, Da = naction.shape  # Get tensor dimensions
        To = nobs['point_cloud'].shape[1]  # obs timesteps
        device = naction.device
        
        # Setup conditioning (same as main compute_loss)
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(B, -1)
            local_cond = None
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            global_cond = None
            local_cond = nobs_features
        
        # Sample points for velocity regularization
        num_time_samples = 16  # Reduced for faster, more consistent compute
        t_samples = torch.rand(B, num_time_samples, device=device)  # (B, 16)
        x_samples = torch.randn(B, num_time_samples, T, Da, device=device)

        a = naction.unsqueeze(1).expand(-1, num_time_samples, -1, -1)  # (B,S,T,Da)
        s_exp = t_samples.view(B, num_time_samples, 1, 1)
        x_gamma = (1 - s_exp) * x_samples + s_exp * a  # on-path samples

        v_list = []
        for i in range(num_time_samples):
            t = t_samples[:, i] * 99.0  # match your model's 0..99 time scale
            v = self.model(x_gamma[:, i], t, local_cond=local_cond, global_cond=global_cond)
            v_list.append(v)
        V = torch.stack(v_list, dim=1)  # (B,S,T,Da)

        # smoothness along s (finite differences across samples)
        Vdiff = V[:, 1:] - V[:, :-1]
        smoothness = torch.mean(Vdiff ** 2)

        # small magnitude control (very small)
        magnitude = torch.mean(V ** 2) * 0.01

        return smoothness + magnitude


# Helper function for dict operations
def dict_apply(data, func):
    """Apply function to all values in nested dict."""
    if isinstance(data, dict):
        return {k: dict_apply(v, func) for k, v in data.items()}
    else:
        return func(data)


class RealRobotFlowPolicy(FlowPolicyWithActionMSE):
    """Real Robot FlowPolicy with dual-view point clouds and enhanced CFM fixes."""
    
    def __init__(self, use_action_mse=True, log_dir=None, use_dual_view=True, **kwargs):
        print("🔧 Initializing RealRobotFlowPolicy with Dual-View point clouds...")
        
        # If using dual view, replace the obs_encoder with DualViewFlowPolicyEncoder
        self.use_dual_view = use_dual_view
        
        super().__init__(use_action_mse=use_action_mse, **kwargs)
        
        # Replace obs_encoder with dual-view encoder if enabled
        if use_dual_view:
            print("🔄 Replacing obs_encoder with DualViewFlowPolicyEncoder...")
            shape_meta = kwargs.get('shape_meta', {})
            obs_shape_meta = shape_meta.get('obs', {})
            obs_dict = {k: v['shape'] for k, v in obs_shape_meta.items()}
            
            encoder_output_dim = kwargs.get('encoder_output_dim', 64)
            pointcloud_encoder_cfg = kwargs.get('pointcloud_encoder_cfg', {})
            use_pc_color = kwargs.get('use_pc_color', True)
            pointnet_type = kwargs.get('pointnet_type', 'mlp')
            
            self.obs_encoder = DualViewFlowPolicyEncoder(
                observation_space=obs_dict,
                out_channel=encoder_output_dim,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                use_pc_color=use_pc_color,
                pointnet_type=pointnet_type,
            )
            
            # Update obs_feature_dim and global_cond_dim
            new_obs_feature_dim = self.obs_encoder.output_shape()
            print(f"   • New obs_feature_dim: {new_obs_feature_dim} (was {self.obs_feature_dim})")
            self.obs_feature_dim = new_obs_feature_dim
            
            # Recreate the model with updated global_cond_dim
            if self.obs_as_global_cond:
                new_global_cond_dim = new_obs_feature_dim * self.n_obs_steps
                print(f"   • New global_cond_dim: {new_global_cond_dim}")
                
                from flow_policy_3d.model.flow.conditional_unet1d import ConditionalUnet1D
                self.model = ConditionalUnet1D(
                    input_dim=self.action_dim,
                    local_cond_dim=None,
                    global_cond_dim=new_global_cond_dim,
                    diffusion_step_embed_dim=kwargs.get('diffusion_step_embed_dim', 128),
                    down_dims=kwargs.get('down_dims', (256, 512, 1024)),
                    kernel_size=kwargs.get('kernel_size', 5),
                    n_groups=kwargs.get('n_groups', 8),
                    condition_type=kwargs.get('condition_type', 'film'),
                    use_down_condition=kwargs.get('use_down_condition', True),
                    use_mid_condition=kwargs.get('use_mid_condition', True),
                    use_up_condition=kwargs.get('use_up_condition', True),
                )
                print("   • Recreated ConditionalUnet1D with new global_cond_dim")
        
        # Setup logging if directory provided
        if log_dir:
            self.setup_logging(log_dir)
        
        print("✅ RealRobotFlowPolicy initialized successfully!")
    
    def set_train(self):
        """Override to add epoch summary hook when switching to eval mode."""
        super().train()
        
    def set_eval(self):
        """Override to print epoch summary when switching to eval mode (end of training epoch)."""
        self.print_epoch_summary()
        super().eval()


class RealRobotDataset(BaseDataset):
    """
    Dataset adapter for real robot data following exact pattern from working code.
    Uses DP-style action normalization like train_flowpolicy_exact_normalized_actions_v2.py
    """
    
    def __init__(self, 
                 zarr_path: str,
                 horizon: int = 4,
                 n_obs_steps: int = 2,
                 n_action_steps: int = 4,
                 train: bool = True,
                 split_ratio: float = 0.9,
                 demo_fraction: float = 1.0,
                 **kwargs):
        self.zarr_path = zarr_path
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.train = train
        self.split_ratio = split_ratio
        self.demo_fraction = demo_fraction

        print(f"📁 Loading zarr dataset from: {zarr_path}")
        
        # Load zarr data and CACHE IN RAM for fast access
        print(f"⚡ Caching dataset in RAM to eliminate I/O bottleneck...")
        z = zarr.open(zarr_path, mode='r')
        
        # Load entire dataset into RAM (this eliminates zarr I/O bottleneck)
        # Load BOTH agentview and eye_in_hand point clouds
        self.agentview_pc_data = np.array(z['data']['agentview_point_cloud'][:]).astype(np.float32)  # (T, N, 6)
        self.eye_in_hand_pc_data = np.array(z['data']['eye_in_hand_point_cloud'][:]).astype(np.float32)  # (T, N, 6)
        self.state_data = np.array(z['data']['state'][:]).astype(np.float32)               # (T, 21)
        self.action_data = np.array(z['data']['action'][:]).astype(np.float32)             # (T, 7)
        self.episode_ends = z['meta']['episode_ends'][:]   # (E,)
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

        # Select a fraction of full demonstrations
        total_demos = len(self.episode_ends)
        if demo_fraction < 1.0:
            num_demos_to_use = max(1, int(total_demos * demo_fraction))
            demo_rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            selected_demo_indices = np.sort(demo_rng.choice(total_demos, num_demos_to_use, replace=False))
            self.episode_starts = self.episode_starts[selected_demo_indices]
            self.episode_ends = self.episode_ends[selected_demo_indices]
            print(f"📉 Using {num_demos_to_use}/{total_demos} demonstrations (demo_fraction={demo_fraction})")
            print(f"   • Selected demo indices: {selected_demo_indices}")
        else:
            num_demos_to_use = total_demos

        print(f"✅ Dataset cached in RAM successfully!")
        print(f"📊 Dataset info:")
        print(f"   • AgentView point cloud shape: {self.agentview_pc_data.shape}")
        print(f"   • Eye-in-Hand point cloud shape: {self.eye_in_hand_pc_data.shape}")
        print(f"   • Actions shape: {self.action_data.shape}")
        print(f"   • Agent pos shape: {self.state_data.shape}")
        print(f"   • Total timesteps: {len(self.action_data)}")
        print(f"   • Demos used: {num_demos_to_use}/{total_demos}")
        print(f"   • Episode lengths: {self.episode_ends - self.episode_starts}")
        
        # Calculate RAM usage
        ram_gb = (self.agentview_pc_data.nbytes + self.eye_in_hand_pc_data.nbytes + self.state_data.nbytes + self.action_data.nbytes) / (1024**3)
        print(f"   • RAM usage: {ram_gb:.2f} GB")

        # Valid sequence starts
        self.indices = []
        for start, end in zip(self.episode_starts, self.episode_ends):
            episode_len = end - start
            if episode_len >= horizon + n_obs_steps - 1:
                max_start = end - horizon - n_obs_steps + 1
                self.indices.extend(range(start, max_start + 1))
        self.indices = np.array(self.indices)

        # Split
        rng = np.random.RandomState(42)
        rng.shuffle(self.indices)
        n_train = int(len(self.indices) * split_ratio)
        self.indices = self.indices[:n_train] if train else self.indices[n_train:]
        self._rng = rng

        print(f"   • Valid indices: {len(self.indices)}")

        if len(self.indices) == 0:
            raise RuntimeError("No sequences available after split; check zarr and split_ratio.")

        # ---- Fit DP action normalizer on a sample of raw actions ----
        sample_indices = rng.choice(self.indices, size=min(2000, len(self.indices)), replace=False)
        action_samples = []
        for idx in sample_indices:
            obs_start = idx
            obs_end = obs_start + self.n_obs_steps
            action_start = obs_end - 1
            action_end = action_start + self.horizon
            a = np.array(self.action_data[action_start:action_end])  # (horizon, 7)
            action_samples.append(a.reshape(-1, a.shape[-1]))
        all_action_sample = np.concatenate(action_samples, axis=0).astype(np.float32)

        # Fit identity action normalizer (NO normalization)
        stat = array_to_stats(all_action_sample)
        self.dp_action_norm: SingleFieldLinearNormalizer = custom_robot_action_normalizer_from_stat(stat)

        # Debug print for identity normalization (should be unchanged)
        a_dp = self.dp_action_norm.encode(all_action_sample)
        pos = a_dp[:, :3]  # position
        rot = a_dp[:, 3:6]  # rotation  
        gripper = a_dp[:, 6:]  # gripper
        print(f"🎯 Identity Action Normalizer (NO normalization):")
        print(f"   • Position range: min={pos.min():.3f}, max={pos.max():.3f}")
        print(f"   • Rotation range: min={rot.min():.3f}, max={rot.max():.3f}")
        print(f"   • Gripper range: min={gripper.min():.3f}, max={gripper.max():.3f}")
        print(f"   • Raw action sample: min={all_action_sample.min():.3f}, max={all_action_sample.max():.3f}")
        
        # Round-trip check (should be zero error for identity)
        rt_err = np.abs(self.dp_action_norm.decode(a_dp) - all_action_sample).mean()
        print(f"   • Round-trip L1 error: {rt_err:.3e} (should be ~0 for identity)")

        # PRE-NORMALIZE ALL ACTIONS and cache in RAM (eliminates worker CPU overhead)
        print(f"⚡ Pre-normalizing all actions to eliminate worker overhead...")
        self.action_data_normalized = self.dp_action_norm.encode(self.action_data).astype(np.float32)
        print(f"✅ All actions pre-normalized and cached!")

        # Storage for FlowPolicy LinearNormalizer (set later in get_normalizer)
        self.normalizer = None

    def get_normalizer(self, mode='limits', **kwargs):
        """
        Fit FlowPolicy's LinearNormalizer on already DP-normalized actions,
        so it becomes identity on 'action'.
        Includes both agentview and eye_in_hand point clouds.
        """
        sample_indices = self._rng.choice(self.indices, size=min(1000, len(self.indices)), replace=False)

        agentview_pc_samples, eye_in_hand_pc_samples, state_samples, action_samples = [], [], [], []
        for idx in sample_indices:
            obs_start = idx
            obs_end = obs_start + self.n_obs_steps

            agentview_pc = np.array(self.agentview_pc_data[obs_start:obs_end])        # (n_obs, N, 6)
            eye_in_hand_pc = np.array(self.eye_in_hand_pc_data[obs_start:obs_end])    # (n_obs, N, 6)
            state = np.array(self.state_data[obs_start:obs_end])  # (n_obs, 21)

            action_start = obs_end - 1
            action_end = action_start + self.horizon
            actions_raw = np.array(self.action_data[action_start:action_end], dtype=np.float32)  # (horizon, 7)
            actions_dp = self.dp_action_norm.encode(actions_raw)

            agentview_pc_samples.append(agentview_pc.reshape(-1, agentview_pc.shape[-1]))
            eye_in_hand_pc_samples.append(eye_in_hand_pc.reshape(-1, eye_in_hand_pc.shape[-1]))
            state_samples.append(state.reshape(-1, state.shape[-1]))
            action_samples.append(actions_dp.reshape(-1, actions_dp.shape[-1]))

        all_agentview_pc = np.concatenate(agentview_pc_samples, axis=0).astype(np.float32)
        all_eye_in_hand_pc = np.concatenate(eye_in_hand_pc_samples, axis=0).astype(np.float32)
        all_state = np.concatenate(state_samples, axis=0).astype(np.float32)
        all_action = np.concatenate(action_samples, axis=0).astype(np.float32)

        # Fit FlowPolicy's normalizer on DP-normalized actions (identity on 'action')
        # Include both point clouds
        normalizer = LinearNormalizer()
        normalizer.fit(data={
            'action': all_action, 
            'agent_pos': all_state, 
            'point_cloud': all_agentview_pc,
            'eye_in_hand_point_cloud': all_eye_in_hand_pc
        }, last_n_dims=1, mode=mode, **kwargs)
        self.normalizer = normalizer
        return normalizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        # Observations
        obs_start = data_idx
        obs_end = obs_start + self.n_obs_steps
        agentview_pc = self.agentview_pc_data[obs_start:obs_end]  # Already float32 from cache
        eye_in_hand_pc = self.eye_in_hand_pc_data[obs_start:obs_end]  # Already float32 from cache
        state = self.state_data[obs_start:obs_end]     # Already float32 from cache

        # Actions (use pre-normalized cache - zero CPU overhead)
        action_start = obs_end - 1
        action_end = action_start + self.horizon
        action_dp = self.action_data_normalized[action_start:action_end]  # Pre-normalized from cache

        return {
            'obs': {
                'point_cloud': torch.from_numpy(agentview_pc).float(),
                'eye_in_hand_point_cloud': torch.from_numpy(eye_in_hand_pc).float(),
                'agent_pos': torch.from_numpy(state).float()
            },
            'action': torch.from_numpy(action_dp).float()
        }
    
    def get_validation_dataset(self):
        """Return validation dataset (shares DP normalizer and FlowPolicy normalizer)."""
        val_dataset = RealRobotDataset(
            zarr_path=self.zarr_path,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            train=False,
            split_ratio=self.split_ratio,
            demo_fraction=self.demo_fraction
        )
        # Share DP action normalizer, normalized actions, & FlowPolicy normalizer if present
        val_dataset.dp_action_norm = self.dp_action_norm
        val_dataset.action_data_normalized = self.action_data_normalized  # Share pre-normalized cache
        val_dataset.normalizer = self.normalizer
        return val_dataset


def create_config():
    """Create configuration for real robot training with dual-view point clouds."""
    shape_meta = {
        'obs': {
            'point_cloud': {'shape': (8192, 6)},  # AgentView XYZ+RGB
            'eye_in_hand_point_cloud': {'shape': (8192, 6)},  # Eye-in-Hand XYZ+RGB
            'agent_pos': {'shape': (21,)}
        },
        'action': {'shape': (7,)}
    }

    cfg = OmegaConf.create({
        'name': 'train_real_robot_dual_view',
        'task_name': 'real_robot_manipulation',
        'shape_meta': shape_meta,
        'exp_name': 'real_robot_agentview_experiment',
        'horizon': 4,
        'n_obs_steps': 2,
        'n_action_steps': 4,
        'n_latency_steps': 0,
        'dataset_obs_steps': 2,
        'keypoint_visible_rate': 1.0,
        'obs_as_global_cond': True,

        'policy': {
            '_target_': '__main__.RealRobotFlowPolicy',  # Uses enhanced FlowPolicy with dual-view and CFM fixes
            'use_action_mse': True,
            'use_dual_view': True,  # Enable dual-view point cloud processing
            'log_dir': '${oc.env:PWD}/FlowPolicy/stacking_cups_dual_view_training_checkpoint/loss_logs',
            'use_point_crop': False,
            'condition_type': 'film',
            'use_down_condition': True,
            'use_mid_condition': True,
            'use_up_condition': True,
            'diffusion_step_embed_dim': 128,
            'down_dims': [256, 512, 1024],
            'crop_shape': [84, 84],
            'encoder_output_dim': 64,
            'horizon': 4,
            'kernel_size': 5,
            'n_action_steps': 4,
            'n_groups': 8,
            'n_obs_steps': 2,
            'obs_as_global_cond': True,
            'shape_meta': shape_meta,
            'use_pc_color': True,
            'pointnet_type': 'mlp',
            'pointcloud_encoder_cfg': {
                'in_channels': 6,
                'out_channels': 64,
                'use_layernorm': True
            },
            'Conditional_ConsistencyFM': {
                'eps': 5e-3,
                'num_segments': 3,
                'boundary': 1,
                'delta': 7e-1,   # Enhanced parameters from working code
                'alpha': 0.8,
                'num_inference_step': 30
            },
            'eta': 0.05
        },

        'task': {
            'dataset': {
                '_target_': '__main__.RealRobotDataset',
                'zarr_path': '/media/carl_ma/MyPassport/stacking_cups/stacking_cups_dual_view.zarr',
                'horizon': 4,
                'n_obs_steps': 2,
                'n_action_steps': 4,
                'demo_fraction': 0.5  # Use 50% of demos (~50 out of 101). Set to 1.0 for all.
            },
            'env_runner': None
        },

        'dataloader': {
            'batch_size': 32,  # Smaller batches for faster loading
            'shuffle': True,
            'num_workers': 16,  # Max workers for parallel loading
            'pin_memory': True,  # Enable for faster CPU->GPU transfer
            'persistent_workers': True,
            'drop_last': True,
            'prefetch_factor': 8  # Aggressive prefetch to buffer data
        },

        'val_dataloader': {
            'batch_size': 32,  # Smaller batches for faster loading
            'shuffle': False,
            'num_workers': 16,  # Max workers for parallel loading
            'pin_memory': True,  # Enable for faster CPU->GPU transfer
            'persistent_workers': True,
            'drop_last': False,
            'prefetch_factor': 8  # Aggressive prefetch to buffer data
        },

        'optimizer': {
            '_target_': 'torch.optim.AdamW',
            'lr': 5e-4,
            'weight_decay': 1e-5
        },

        'training': {
            'seed': 42,
            'device': 'cuda',
            'num_epochs': 400,
            'gradient_accumulate_every': 4,  # Accumulate 4 batches = effective batch 128
            'lr_scheduler': 'cosine',
            'lr_warmup_steps': 100,
            'use_ema': True,
            'rollout_every': 999,
            'checkpoint_every': 5,
            'val_every': 1,
            'sample_every': 5,
            'debug': False,
            'max_train_steps': None,
            'max_val_steps': None,
            'tqdm_interval_sec': 1.0,
            'resume': True,  # Set to True to resume from latest checkpoint
            'use_amp': True,  # Enable automatic mixed precision for faster training
            'compile_model': False  # Set to True if using PyTorch 2.0+ for additional speedup
        },

        'ema': {
            '_target_': 'flow_policy_3d.model.flow.ema_model.EMAModel',
            'update_after_step': 0,
            'inv_gamma': 1.0,
            'power': 0.75,
            'min_value': 0.0,
            'max_value': 0.9999
        },

        'checkpoint': {
            'topk': {
                'k': 1,
                'mode': 'max',
                'monitor_key': 'test_mean_score'
            },
            'save_ckpt': True,
            'save_last_ckpt': True,
            'save_last_snapshot': False
        },

        'logging': {
            'project': 'flowpolicy_real_robot',
            'group': 'dual_view',
            'name': 'real_robot_training',
            'mode': 'disabled'
        }
    })
    return cfg


if __name__ == "__main__":
    # Import FlowPolicy training workspace


    # Create config
    cfg = create_config()

    # Create output directory
    output_dir = pathlib.Path(__file__).parent / 'stacking_cups_dual_view_training_checkpoint'
    output_dir.mkdir(exist_ok=True)

    print("🚀 Starting Real Robot FlowPolicy Training with DUAL-VIEW Point Clouds...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Training for {cfg.training.num_epochs} epochs")
    print(f"📊 Batch size: {cfg.dataloader.batch_size}")
    print(f"🔧 Debug mode: {cfg.training.debug}")
    print(f"📈 Logging mode: {cfg.logging.mode}")
    print("🔧 TRAINING CONFIGURATION:")
    print("   ✅ Identity Action Normalizer - NO normalization (raw actions)")
    print("   ✅ FlowPolicy LinearNormalizer - Identity on raw actions")
    print("   ✅ DUAL-VIEW: AgentView + Eye-in-Hand point clouds")
    print("   ✅ Separate PointNet encoders with concatenated features")
    print("   ✅ Same dataset split approach")

    # Create workspace and run training
    workspace = TrainFlowPolicyWorkspace(cfg, output_dir=str(output_dir))
    workspace.run()