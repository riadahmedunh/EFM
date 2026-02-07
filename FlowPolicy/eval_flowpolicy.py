#!/usr/bin/env python3
"""
Flow Matching Policy rollout with EXACT environment construction and hand-eye rendering.
Combines the exact rendering from dual_view_renderer.py with flow policy prediction from eval_flowpolicy.py
and follows the rollout pattern from policy_rollout_env_switch.py
"""

# ============================== COMMON IMPORTS (EXACT from working code) ==============================
import os
import sys
import argparse
import numpy as np
import copy as pycopy
import shutil
import time
import json
import glob
import warnings
warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm is deprecated.*")

import torch
import torch.nn as nn
import torch.nn.functional as F

import open3d as o3d
import cv2
import h5py
from PIL import Image
import torchvision.transforms as T
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import collections
import inspect
import dill

# --- 3DGS / Feature Splatting imports ---
sys.path.append(".")
sys.path.append("feature-splatting-inria")
sys.path.append("fov_module")
sys.path.append("recog_module")
sys.path.append("grasp_pose_module")
sys.path.append("diffusion_policy")
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel, skip_feat_decoder
import featsplat_editor
from gaussian_edit import edit_utils

# --- Advanced segmentation imports (MaskCLIP) ---
import maskclip_onnx

# --- 3DGS (rollout uses these) ---
from gaussian_renderer import render, GaussianModel as GR_GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scipy.spatial.transform import Rotation as R_scipy
from utils.general_utils import safe_state

# --- Robotics ---
from spatialmath import SE3, SO3
import roboticstoolbox as rtb
from roboticstoolbox.robot.ERobot import ERobot

# --- Flow Policy specific imports ---
sys.path.append('/home/carl_lab/Riad/graspbility/FlowPolicy/FlowPolicy')
sys.path.append('/home/carl_lab/Riad/graspbility/FlowPolicy/FlowPolicy/flow_policy_3d')
from train_real_robot_flowpolicy import RealRobotDataset, create_model
from flow_policy_3d.common.pytorch_util import dict_apply

# Recognition imports
from recognizability_core import DualViewRecognizer

# ============================== CONSTANTS (EXACT from working code) ==============================

# Static hand->camera transform (EXACT)
gTc = np.array([
    [ 0.21166931, -0.9773153,  -0.00713539,  0.04742457],
    [ 0.97400948,  0.21154441, -0.08095982,  0.02245319],
    [ 0.08063272,  0.01018677,  0.99669183, -0.0504773 ],
    [ 0.0,         0.0,         0.0,         1.0]
], dtype=np.float64)

# Target objects & colors (EXACT)
OBJECTS_TO_SEGMENT = {
    "blue_bottle":  {"query": "a blue candy", "color": [0.1, 0.3, 0.9], "neg_query": "red candy, brach's candy"},
    "red_candy":    {"query": "a red candy bar",  "color": [0.9, 0.2, 0.2], "neg_query": "blue candy, brach's candy"},
    "brachs_candy": {"query": "a Brach's candy in a green and yellow wrapper","color": [0.2, 0.9, 0.2], "neg_query": "blue candy, red candy"},
}

# Constants (EXACT from working code)
DEF_FRONT_MARKER_HEIGHT = 1.2
DEF_CAMERA_OFFSET = 1.0
DEF_CAMERA_TILT_DEGREES = -30.0
DEF_FRUSTUM_SCALE = 0.22
DEF_WALL_HEIGHT, DEF_V_MIN, DEF_U_MARGIN, DEF_THICKNESS, DEF_OUTSET = 1.0, -0.005, 0.010, 0.010, 0.001
PLANE_PLY_PATH = "./gaussian/black_plane_front.ply"
Z_CENTER_MODE = "median"
Z_BAND_ABS = 0.010
Z_BAND_MAD_K = 3.0
Z_BAND_MAX = 0.050

# Robot configuration (EXACT from reference)
INIT_QPOS=[0.0,0.1,0.0,-2.15,0.0,2.25,0.78,0.04,0.04]

def load_npz_intrinsics(npz_file):
    d = np.load(npz_file)
    fx, fy, cx, cy = float(d["fx"]), float(d["fy"]), float(d["ppx"]), float(d["ppy"])
    W, H = int(cx * 2), int(cy * 2)
    FoVx = 2 * np.arctan(W / (2 * fx))
    FoVy = 2 * np.arctan(H / (2 * fy))
    return W, H, FoVx, FoVy

def mask_points_in_rotated_rect_xy(xy: np.ndarray, center_xy: np.ndarray, size_wh: np.ndarray, angle_rad: float) -> np.ndarray:
    cx,cy=center_xy; w,h=size_wh
    xy0=xy-np.array([cx,cy],dtype=np.float32)[None,:]
    ca,sa=np.cos(-angle_rad),np.sin(-angle_rad)
    R2=np.array([[ca,-sa],[sa,ca]],dtype=np.float32)
    uv=xy0@R2.T
    return (np.abs(uv[:,0])<=w*0.5)&(np.abs(uv[:,1])<=h*0.5)

# --- EXACT camera wrappers from reference code ---
class HandEyeDummyCamera:
    def __init__(self, R, T, FoVx, FoVy, W, H):
        self.projection_matrix = getProjectionMatrix(0.01, 100.0, FoVx, FoVy).T.cuda()
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.zeros(3), 1.0)).T.cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3,:3]
        self.image_width, self.image_height = W, H
        self.FoVx, self.FoVy = FoVx, FoVy

class AgentDummyCamera:
    def __init__(self, R, T, FoVx, FoVy, W, H):
        self.projection_matrix = getProjectionMatrix(0.01, 100.0, FoVx, FoVy).transpose(0,1).cuda()
        self.world_view_transform = torch.tensor(getWorld2View2(R, T), dtype=torch.float32).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3,:3]
        self.image_width, self.image_height = W, H
        self.FoVx, self.FoVy = FoVx, FoVy

class GaussianSplatRenderer:
    """EXACT hand-eye renderer from reference code with depth support."""
    def __init__(self, model_path, source_path, invert=False):
        self.gTc = np.array([[0.21166931,-0.9773153,-0.00713539,0.04742457],
                             [0.97400948,0.21154441,-0.08095982,0.02245319],
                             [0.08063272,0.01018677,0.99669183,-0.0504773],
                             [0,0,0,1]], np.float32)
        if invert: self.gTc = np.linalg.inv(self.gTc)
        p = argparse.ArgumentParser(add_help=False)
        mp = ModelParams(p, sentinel=True); pp = PipelineParams(p)
        a,_ = p.parse_known_args(['--model_path',model_path,'--source_path',model_path])
        self.pipeline = pp.extract(a)
        ds = mp.extract(a)
        bg = [1,1,1] if ds.white_background else [0,0,0]
        self.background = torch.tensor(bg, dtype=torch.float32, device="cuda")
        intr = os.path.join(source_path, "rgb_intrinsics.npz")
        self.W, self.H, self.FoVx, self.FoVy = load_npz_intrinsics(intr)

    def render(self, bTg, scene):
        cTb  = np.linalg.inv(bTg @ self.gTc)
        Rcw  = cTb[:3,:3].T
        tcw  = cTb[:3,3]
        cam  = HandEyeDummyCamera(Rcw, tcw, self.FoVx, self.FoVy, self.W, self.H)
        with torch.no_grad():
            return render(cam, scene, self.pipeline, self.background)["render"]
    
    def render_with_depth(self, bTg, scene):
        """Render both RGB and depth images"""
        cTb  = np.linalg.inv(bTg @ self.gTc)
        Rcw  = cTb[:3,:3].T
        tcw  = cTb[:3,3]
        cam  = HandEyeDummyCamera(Rcw, tcw, self.FoVx, self.FoVy, self.W, self.H)
        with torch.no_grad():
            out = render(cam, scene, self.pipeline, self.background)
            return out["render"], out.get("render_depth", None)

# ---- ROI & agent-view helpers (exact math)
def find_latest_roi_json(base_dir: str) -> str | None:
    pats = [os.path.join(base_dir,"*_table_roi_rot_*.json"),
            os.path.join(base_dir,"*_table_roi_*.json"),
            os.path.join(base_dir,"*roi*.json")]
    cands=[]
    for p in pats: cands.extend(glob.glob(p))
    if not cands: return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def parse_roi(roi_path: str):
    with open(roi_path, "r") as f: data = json.load(f)
    result_dict={}; corners3d=None
    if "rotated_rect" in data:
        rr = data["rotated_rect"]; z = float(rr.get("z_level", 0.0))
        corners_xy = np.asarray(rr["corners_xy"], dtype=np.float32)
        corners3d = np.c_[corners_xy, np.full((4,), z, np.float32)]
        result_dict = dict(kind="rot",
            center_xy=np.asarray(rr["center_xy"],dtype=np.float32),
            size_wh=np.asarray(rr["size_wh"],dtype=np.float32),
            angle_rad=float(rr["angle_rad"]), z_level=z,
            corners3d=corners3d.astype(np.float32))
    elif "corners_xyz" in data:
        corners3d = np.asarray(data["corners_xyz"], dtype=np.float32)
        z = float(corners3d[:,2].mean())
        result_dict = dict(kind="aa", z_level=z, corners3d=corners3d.astype(np.float32))
    else: raise ValueError(f"Unrecognized ROI JSON schema in {roi_path}")
    C_xy = corners3d[:,:2].mean(axis=0)
    ang = np.arctan2(corners3d[:,1]-C_xy[1], corners3d[:,0]-C_xy[0])
    sorted_corners = corners3d[np.argsort(ang)]
    result_dict["corners3d"] = sorted_corners
    result_dict['center3d'] = sorted_corners.mean(axis=0).astype(np.float32)
    return result_dict

def ordered_edges(corners3d:np.ndarray):
    return [(corners3d[0],corners3d[1]),
            (corners3d[1],corners3d[2]),
            (corners3d[2],corners3d[3]),
            (corners3d[3],corners3d[0])]

def robust_z_band(z_vals: np.ndarray, z_center_mode: str, roi_z: float) -> tuple[float, float]:
    z_vals = np.asarray(z_vals, dtype=np.float32)
    if z_vals.size == 0: return float(roi_z), float(0.010)
    z_med = float(np.median(z_vals))
    mad   = float(np.median(np.abs(z_vals - z_med)))
    sigma_hat = 1.4826 * mad
    half_band = max(Z_BAND_ABS, Z_BAND_MAD_K * sigma_hat)
    if Z_BAND_MAX is not None: half_band = min(half_band, Z_BAND_MAX)
    z_center = float(roi_z) if z_center_mode == "roi" else z_med
    return z_center, half_band

def table_normal_from_roi(env_pts:np.ndarray,corners3d:np.ndarray,z_level:float,z_band=0.03):
    x,y=corners3d[:,0],corners3d[:,1]
    xmin,xmax,ymin,ymax=x.min(),x.max(),y.min(),y.max()
    m_xy=(env_pts[:,0]>=xmin)&(env_pts[:,0]<=xmax)&(env_pts[:,1]>=ymin)&(env_pts[:,1]<=ymax)
    m_z=np.abs(env_pts[:,2]-z_level)<=z_band
    P = env_pts[m_xy & m_z]
    if P.shape[0]<200: n=np.array([0,0,1],np.float32)
    else:
        c=P.mean(axis=0); X=P-c; _,_,vh=np.linalg.svd(X,full_matrices=False)
        n=vh[-1]; n=n/(np.linalg.norm(n)+1e-12)
    if n[2]<0: n=-n
    return n

def get_camera_matrices(position: np.ndarray, target: np.ndarray, world_up: np.ndarray, tilt_degrees: float = 0.0):
    forward_parallel = target - position
    forward_parallel -= np.dot(forward_parallel, world_up) * world_up
    z_axis_level = forward_parallel / (np.linalg.norm(forward_parallel) + 1e-12)
    x_axis_level = np.cross(z_axis_level, world_up); x_axis_level /= (np.linalg.norm(x_axis_level) + 1e-12)
    y_axis_level = np.cross(z_axis_level, x_axis_level)
    R_level = np.column_stack([x_axis_level, y_axis_level, z_axis_level])
    tilt_rad = np.deg2rad(tilt_degrees); tilt_rotation = R_scipy.from_rotvec(tilt_rad * x_axis_level).as_matrix()
    R_final_pose = tilt_rotation @ R_level
    C_pose = np.eye(4); C_pose[:3, :3] = R_final_pose; C_pose[:3, 3] = position
    V_view = np.linalg.inv(C_pose)
    return C_pose, V_view

# =============================================================================
# 3DGS helpers
# =============================================================================
def load_gaussian_model_from_ply(path):
    p=argparse.ArgumentParser(add_help=False); ModelParams(p)
    args,_=p.parse_known_args(['--source_path','dummy','--distill_feature_dim','0'])
    g=GR_GaussianModel(args.sh_degree,args.distill_feature_dim); g.load_ply(path)
    return g

def transform_gaussians(g,M):
    dev=g._xyz.device; out=pycopy.deepcopy(g)
    T=torch.tensor(M,dtype=torch.float32,device=dev)
    ph=torch.cat([g._xyz,torch.ones(g._xyz.shape[0],1,device=dev)],1)
    pos=(T@ph.T).T[:,:3]
    Rm=T[:3,:3].cpu().numpy()
    quat=R_scipy.from_matrix(Rm).as_quat()  # xyzw
    qt=torch.tensor([quat[3],quat[0],quat[1],quat[2]],device=dev)  # wxyz
    w0,x0,y0,z0=qt/qt.norm(); w1,x1,y1,z1=g._rotation.T
    nw=w0*w1-x0*x1-y0*y1-z0*z1
    nx=w0*x1+x0*w1+y0*z1-z0*y1
    ny=w0*y1-x0*z1+y0*w1+z0*x1
    nz=w0*z1+x0*y1-y0*x1+z0*w1
    rot=torch.stack([nw,nx,ny,nz],1)
    out._xyz=torch.nn.Parameter(pos); out._rotation=torch.nn.Parameter(rot)
    return out

def compose_gaussians(mlist):
    if not mlist: return GR_GaussianModel(0,0)
    valid=[m for m in mlist if hasattr(m,'_xyz') and m._xyz.shape[0]>0]
    if not valid: return GR_GaussianModel(0,0)
    base=pycopy.deepcopy(valid[0])
    for a in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']:
        if hasattr(base, a):
            data=[getattr(m,a).data for m in valid if hasattr(m, a)]
            if data: setattr(base,a,torch.nn.Parameter(torch.cat(data,0)))
    return base

def robust_to_pil(img):
    arr = np.asarray(img)
    if arr.ndim != 3: raise ValueError(f"Unsupported image shape {arr.shape}")
    # CHW->HWC
    if arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4):
        arr = np.transpose(arr, (1,2,0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr*255.0,0,255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr,0,255).astype(np.uint8)
    if arr.shape[-1]==1: arr = np.repeat(arr,3,axis=-1)
    elif arr.shape[-1]==4: arr = arr[:,:,:3]
    from PIL import Image
    return Image.fromarray(arr, mode="RGB")

class FrankaEmikaPanda(ERobot):
    def __init__(self):
        urdf=os.path.expanduser('~/Riad/graspbility/GraspSplats/urdf/panda.urdf')
        if not os.path.exists(urdf): raise FileNotFoundError(urdf)
        links,name,_,_=self.URDF_read(urdf)
        super().__init__(links,name=name)
        self.default_joint_pos=np.array(INIT_QPOS)
        self.addconfiguration('qr',self.default_joint_pos)

class FrankaGaussianEnv:
    """
    EXACT environment class from reference code with dual scene management.
    """
    def __init__(self, env_model_path, robot_links_dir,
                 source_path, img_w, img_h, invert_hand_eye=False,
                 z_scene_offset=0.0, z_base_offset=0.0,
                 x_base_offset=0.0, y_base_offset=0.0,
                 # agent-view placement:
                 fov_y_deg=55.0, outset=DEF_OUTSET, camera_offset=DEF_CAMERA_OFFSET,
                 front_marker_height=DEF_FRONT_MARKER_HEIGHT, camera_tilt=DEF_CAMERA_TILT_DEGREES):
        self.z_scene_offset=z_scene_offset
        self.z_base_offset=z_base_offset
        self.x_base_offset=x_base_offset
        self.y_base_offset=y_base_offset

        ply=os.path.join(env_model_path,"point_cloud","iteration_15000","point_cloud.ply")
        self.unfiltered_env_gaussians = load_gaussian_model_from_ply(ply)

        # robot kinematics
        self.robot_kin=FrankaEmikaPanda()
        self.robot_kin.q=self.robot_kin.default_joint_pos.copy()

        ee_cands=['panda_hand_tcp','panda_hand','panda_link8','link8','tool0']
        self.end_link=None
        if hasattr(self.robot_kin,'link_dict'):
            for n in ee_cands:
                if n in self.robot_kin.link_dict:
                    self.end_link=self.robot_kin.link_dict[n]; break
        if self.end_link is None: self.end_link=self.robot_kin.ee_links[0]

        self.ets_arm=self.robot_kin.ets(end=self.end_link)
        self.arm_jindices=np.array(self.ets_arm.jindices,dtype=int)

        # link meshes
        self.link_models={}
        for i in range(10):
            f=os.path.join(robot_links_dir,f"link{i}_default.ply")
            if os.path.exists(f): self.link_models[f"link{i}"]=load_gaussian_model_from_ply(f)
        self.ply_to_kin={"link0":1,"link1":2,"link2":3,"link3":4,
                         "link4":5,"link5":6,"link6":7,"link7":8,
                         "link8":11,"link9":12}

        # hand-eye renderer (EXACT from reference)
        self.renderer = GaussianSplatRenderer(env_model_path, source_path, invert=invert_hand_eye)
        self.disp_W, self.disp_H = img_w, img_h

        # ----- build AGENT-VIEW camera from ROI and create the FILTERED scene -----
        base_dir = os.path.dirname(ply)
        roi_path = find_latest_roi_json(base_dir) or find_latest_roi_json(env_model_path)
        if roi_path is None: raise FileNotFoundError(f"No ROI json found near {base_dir}")
        roi = parse_roi(roi_path)

        # filter for agent view scene
        original_pts = self.unfiltered_env_gaussians._xyz.detach().cpu().numpy().astype(np.float32)
        xy = original_pts[:, :2]
        if roi["kind"] == "rot":
            mask_xy = mask_points_in_rotated_rect_xy(xy, roi["center_xy"], roi["size_wh"], roi["angle_rad"])
        else:
            x_c, y_c = roi["corners3d"][:, 0], roi["corners3d"][:, 1]
            xmin, xmax, ymin, ymax = x_c.min(), x_c.max(), y_c.min(), y_c.max()
            mask_xy = (xy[:,0]>=xmin)&(xy[:,0]<=xmax)&(xy[:,1]>=ymin)&(xy[:,1]<=ymax)
        z_center, half_band = robust_z_band(original_pts[mask_xy, 2], Z_CENTER_MODE, roi["z_level"])
        mask_z = np.abs(original_pts[:,2]-z_center)<=half_band
        final_mask = mask_xy & mask_z

        self.env_gaussians = GR_GaussianModel(sh_degree=self.unfiltered_env_gaussians.active_sh_degree, distill_feature_dim=0)
        for attr in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','max_radii2D','_features_extra']:
            if hasattr(self.unfiltered_env_gaussians, attr):
                t = getattr(self.unfiltered_env_gaussians, attr)
                if t is None: continue
                if isinstance(t, torch.nn.Parameter): t = t.data
                if t.shape[0] > 0: setattr(self.env_gaussians, attr, torch.nn.Parameter(t[final_mask]))

        # table normal for agent camera
        up_world = table_normal_from_roi(self.env_gaussians._xyz.detach().cpu().numpy(), roi["corners3d"], roi["z_level"])

        edges = ordered_edges(roi["corners3d"]); candidates=[]
        for (a,b) in edges:
            edge_vec=(b-a)
            t_edge=edge_vec/(np.linalg.norm(edge_vec)+1e-12)
            mid=(a+b)*0.5
            s_normal=np.cross(up_world,t_edge); s_normal/=(np.linalg.norm(s_normal)+1e-12)
            if np.dot(mid-roi["center3d"], s_normal)<0:
                t_edge*=-1; s_normal*=-1
            candidates.append(dict(mid=mid,s_normal=s_normal))

        robot_base_pos=np.zeros(3)
        c2r=robot_base_pos-roi["center3d"]; c2r-=np.dot(c2r,up_world)*up_world
        v_robot_plane=c2r/(np.linalg.norm(c2r)+1e-12)
        front_wall=max(candidates,key=lambda c:np.dot(c["s_normal"],-v_robot_plane))

        front_point = front_wall["mid"] + front_marker_height * up_world
        render_camera_pos = front_point + front_wall["s_normal"] * camera_offset
        C_agent, V_agent = get_camera_matrices(render_camera_pos, roi["center3d"], up_world, tilt_degrees=camera_tilt)

        FoVy = np.deg2rad(fov_y_deg); FoVx = 2*np.arctan(np.tan(FoVy/2)*(img_w/img_h))
        self.agent_cam = AgentDummyCamera(R=V_agent[:3,:3].T, T=V_agent[:3,3], FoVx=FoVx, FoVy=FoVy, W=img_w, H=img_h)
        self.agent_pose_C = C_agent
        self.last_gripper_state=0.0

    def _fkine_offset(self,q,end=None):
        T= self.robot_kin.fkine(q,end=end).A.copy()
        T[0,3]-=self.x_base_offset; T[1,3]-=self.y_base_offset; T[2,3]-=self.z_base_offset
        return SE3(T)

    def _fkine_all_offset(self,Q):
        Ts=self.robot_kin.fkine_all(Q)
        for i,T in enumerate(Ts):
            A=T.A.copy()
            A[0,3]-=self.x_base_offset; A[1,3]-=self.y_base_offset; A[2,3]-=self.z_base_offset
            Ts[i]=SE3(A)
        return Ts

    def reset_to_qpos(self,qpos):
        assert len(qpos)==9
        self.robot_kin.q=qpos.copy()
        self.last_gripper_state=float(qpos[7]*2.0)
        return self._get_obs(),{}

    def _ik_to_pose(self, Tgoal: SE3) -> bool:
        q0 = self.robot_kin.q[:7].copy()
        sig = inspect.signature(self.robot_kin.ikine_LM); kw={}
        if 'ilimit' in sig.parameters and 'slimit' in sig.parameters: kw=dict(ilimit=100, slimit=2000)
        elif 'ilim' in sig.parameters and 'slim' in sig.parameters: kw=dict(ilim=100, slim=2000)
        sol = self.robot_kin.ikine_LM(Tgoal, end=self.end_link, q0=q0, mask=[1]*6, joint_limits=False, tol=1e-8, **kw)
        if sol.success and np.all(np.isfinite(sol.q)):
            self.robot_kin.q[:7]=sol.q; return True
        return False

    def step_delta_tool_axisangle(self, dp_tool, rvec_tool, grip_width_m):
        Tep = self.robot_kin.fkine(self.robot_kin.q, end=self.end_link)
        R_new = Tep.R * SO3.Exp(rvec_tool)
        p_new = Tep.t + (Tep.R @ dp_tool)
        self._ik_to_pose(SE3.Rt(R_new, p_new))
        fj = float(grip_width_m) * 0.5
        self.robot_kin.q[7]=self.robot_kin.q[8]=fj
        self.last_gripper_state=float(grip_width_m)
        return self._get_obs(),{}

    def _render_handeye(self, scene):
        """EXACT hand-eye rendering from reference code"""
        eeA=self._fkine_offset(self.robot_kin.q,end=self.end_link).A
        img_t = self.renderer.render(eeA, scene)
        img8  = cv2.cvtColor((img_t.permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # crop to match dataset
        top_pct, bottom_pct, left_pct, right_pct = 0.05, 0.10, 0.10, 0.10
        h,w = img8.shape[:2]
        y0=int(round(h*top_pct)); y1=h-int(round(h*bottom_pct))
        x0=int(round(w*left_pct)); x1=w-int(round(w*right_pct))
        y0,y1=max(0,y0),min(h,y1); x0,x1=max(0,x0),min(w,x1)
        if y1>y0 and x1>x0: img8 = img8[y0:y1, x0:x1]
        img8  = cv2.resize(img8, (self.disp_W, self.disp_H))
        return img8.transpose(2,0,1).astype(np.float32)/255.0  # CHW RGB

    def _render_handeye_with_depth(self, scene):
        """Hand-eye rendering with depth for point cloud generation"""
        eeA=self._fkine_offset(self.robot_kin.q,end=self.end_link).A
        img_t, depth_t = self.renderer.render_with_depth(eeA, scene)
        
        # Process RGB image
        img8  = cv2.cvtColor((img_t.permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # crop to match dataset
        top_pct, bottom_pct, left_pct, right_pct = 0.05, 0.10, 0.10, 0.10
        h,w = img8.shape[:2]
        y0=int(round(h*top_pct)); y1=h-int(round(h*bottom_pct))
        x0=int(round(w*left_pct)); x1=w-int(round(w*right_pct))
        y0,y1=max(0,y0),min(h,y1); x0,x1=max(0,x0),min(w,x1)
        if y1>y0 and x1>x0: 
            img8 = img8[y0:y1, x0:x1]
        img8 = cv2.resize(img8, (self.disp_W, self.disp_H))
        rgb_chw = img8.transpose(2,0,1).astype(np.float32)/255.0  # CHW RGB
        
        # Process depth image with same cropping and resize
        if depth_t is not None:
            depth_np = depth_t.squeeze().cpu().numpy()  # Remove channel dim if present
            if y1>y0 and x1>x0:
                depth_np = depth_np[y0:y1, x0:x1]
            depth_np = cv2.resize(depth_np, (self.disp_W, self.disp_H), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback depth if render doesn't support depth
            depth_np = np.ones((self.disp_H, self.disp_W), dtype=np.float32) * 0.5
            
        return rgb_chw, depth_np

    def _render_agentview(self, scene):
        with torch.no_grad():
            img_t = render(self.agent_cam, scene, self.renderer.pipeline, self.renderer.background)["render"]
        img8 = (img_t.permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)
        img8 = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
        if (img8.shape[1], img8.shape[0]) != (self.disp_W, self.disp_H):
            img8 = cv2.resize(img8, (self.disp_W, self.disp_H))
        return img8.transpose(2,0,1).astype(np.float32)/255.0  # CHW RGB

    def _get_obs(self):
        """EXACT observation generation from reference code with depth-based point clouds"""
        poses = self._fkine_all_offset(self.robot_kin.q)
        rlinks=[]
        for name,m in self.link_models.items():
            idx=self.ply_to_kin.get(name)
            if idx: rlinks.append(transform_gaussians(m,poses[idx].A))
        robot_g=compose_gaussians(rlinks)

        agent_env = pycopy.deepcopy(self.env_gaussians)
        agent_env._xyz.data += torch.tensor([0,0,self.z_scene_offset], device=agent_env._xyz.device)
        agent_scene = compose_gaussians([agent_env, robot_g])

        hand_env = pycopy.deepcopy(self.unfiltered_env_gaussians)
        hand_env._xyz.data += torch.tensor([0,0,self.z_scene_offset], device=hand_env._xyz.device)
        hand_scene = compose_gaussians([hand_env, robot_g])

        # Render hand-eye view with depth
        hand_bgr_chw, hand_depth = self._render_handeye_with_depth(hand_scene)
        agent_bgr_chw = self._render_agentview(agent_scene)

        eeA = self._fkine_offset(self.robot_kin.q, end=self.end_link).A
        return {
            'ee_states': eeA.flatten().astype(np.float32),
            'joint_states': self.robot_kin.q[:7].astype(np.float32),
            'eye_in_hand_rgb': hand_bgr_chw[[2,1,0], :, :],  # RGB
            'eye_in_hand_depth': hand_depth,                 # Depth for point cloud generation
            'agentview': agent_bgr_chw[[2,1,0], :, :],       # RGB
            'gripper_states': np.array([self.last_gripper_state], dtype=np.float32)
        }

    def close(self): pass

# =============================================================================
# Demo I/O + initialization (EXACT from reference code)
# =============================================================================
def list_demos(f):
    return sorted(int(k.split("_")[-1]) for k in f["data"].keys() if k.startswith("demo_"))

def pose_vec_to_4x4_matrix(pose_vec):
    if pose_vec.shape[0] != 16: raise ValueError(f"Pose vector must have 16 elements, got {pose_vec.shape[0]}")
    return pose_vec.reshape((4, 4)).T

def pose16_to_T_transposed(pose_flat16):
    T = np.asarray(pose_flat16, dtype=np.float64).reshape(4,4).T
    T[3,:] = [0,0,0,1]
    R = T[:3,:3]; U,S,Vt = np.linalg.svd(R); R = U @ Vt
    if np.linalg.det(R) < 0: U[:,-1] *= -1; R = U @ Vt
    T[:3,:3] = R; return T

def load_demo_init(path, demo_index):
    with h5py.File(path,"r") as f:
        demos = list_demos(f)
        if demo_index not in demos: raise ValueError(f"demo_{demo_index} not found; available={demos}")
        g = f[f"data/demo_{demo_index}"]; init = {}; obs = g["obs"]
        if "joint_states" in obs:
            q7 = obs["joint_states"][0].astype(np.float64)
            q9 = np.zeros(9, dtype=np.float64); q9[:7]=q7; init["q0"] = q9
        if "ee_states" in obs: init["A0"] = pose16_to_T_transposed(obs["ee_states"][0])
        if "gripper_states" in obs: init["grip0"] = float(obs["gripper_states"][0][0])
        return init

def initialize_env(env, init):
    if "q0" in init:
        q = init["q0"].copy()
        if "grip0" in init:
            g = float(init["grip0"]) * 0.5
            q[7]=q[8]=g; env.last_gripper_state=float(g*2.0)
        env.reset_to_qpos(q)
    elif "A0" in init:
        A0 = init["A0"]
        env._ik_to_pose(SE3.Rt(A0[:3,:3], A0[:3,3], check=False))
        if "grip0" in init:
            g = float(init["grip0"]) * 0.5
            env.robot_kin.q[7]=env.robot_kin.q[8]=g
            env.last_gripper_state=float(g*2.0)

# =============================================================================
# Flow Policy Action Predictor (using real FlowPolicy model)
# =============================================================================
class FlowPolicyEvaluator:
    """EXACT FlowPolicy evaluator from eval_flowpolicy.py with prediction capability"""
    def __init__(self, checkpoint_path, dataset_path, device='cuda'):
        self.device = device
        self.dataset_path = dataset_path
        
        print(f"[FLOW] Loading dataset from {dataset_path}")
        self.dataset = RealRobotDataset(
            hdf5_path=dataset_path,
            n_obs_steps=2,
            horizon=4,
            n_action_steps=4
        )
        
        print(f"[FLOW] Loading model from {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Load normalization stats
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.norm_stats = checkpoint.get('normalization_stats', {})
        
    def _load_model(self, checkpoint_path):
        """Load the trained FlowPolicy model."""
        # Create model with same architecture
        model = create_model(self.dataset, device=self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
        
    def _denormalize_actions(self, normalized_actions):
        """Denormalize actions using stored statistics."""
        if 'action_mean' in self.norm_stats and 'action_std' in self.norm_stats:
            action_mean = torch.FloatTensor(self.norm_stats['action_mean']).to(self.device)
            action_std = torch.FloatTensor(self.norm_stats['action_std']).to(self.device)
            return normalized_actions * action_std + action_mean
        else:
            print("[FLOW] Warning: No action normalization stats found. Returning normalized actions.")
            return normalized_actions
    
    def compute_delta_pose(self, actions):
        """Compute delta pose from action sequence.
        
        Assumes actions are [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
        """
        # Sum up the deltas over the horizon (actions shape: [horizon, 7])
        cumulative_delta = torch.sum(actions, dim=0)  # Sum over time dimension
        
        # Extract pose deltas (assuming first 6 dimensions are pose)
        delta_pose = cumulative_delta[:6]  # [dx, dy, dz, drx, dry, drz]
        gripper_action = actions[-1, 6]  # Final gripper action
        
        return delta_pose, gripper_action

    def _depth_to_pointcloud(self, depth_image, max_points=1024):
        """Convert depth image to 3D point cloud using EXACT same method as training dataset.
        
        Args:
            depth_image: numpy array of shape (H, W) with depth values
            max_points: maximum number of points to sample
            
        Returns:
            point_cloud: numpy array of shape (max_points, 3) with XYZ coordinates
        """
        # Camera intrinsics for hand-eye camera (EXACT same values as training dataset)
        height, width = depth_image.shape
        fx = width * 0.8  # ~512 for 640 width
        fy = height * 0.8  # ~384 for 480 height  
        cx = width / 2.0   # center x
        cy = height / 2.0  # center y
        
        # Convert depth values - depth from Gaussian renderer is already in meters
        depth_m = depth_image.astype(np.float32)
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Only process pixels with valid depth (non-zero and finite)
        valid_mask = (depth_m > 0) & np.isfinite(depth_m)
        valid_u = u[valid_mask]
        valid_v = v[valid_mask]
        valid_depth = depth_m[valid_mask]
        
        # Convert to 3D points
        x = (valid_u - cx) * valid_depth / fx
        y = (valid_v - cy) * valid_depth / fy
        z = valid_depth
        
        # Stack into point cloud
        points = np.stack([x, y, z], axis=-1)  # Shape: (N, 3)
        
        # Sample or pad to get exactly max_points
        num_points = points.shape[0]
        
        if num_points >= max_points:
            # Randomly sample max_points
            indices = np.random.choice(num_points, max_points, replace=False)
            point_cloud = points[indices]
        else:
            # Pad with zeros or repeat points
            if num_points > 0:
                # Repeat points to fill max_points
                repeat_factor = max_points // num_points
                remainder = max_points % num_points
                
                repeated_points = np.tile(points, (repeat_factor, 1))
                if remainder > 0:
                    extra_points = points[:remainder]
                    point_cloud = np.vstack([repeated_points, extra_points])
                else:
                    point_cloud = repeated_points
            else:
                # No valid depth points, create zeros
                point_cloud = np.zeros((max_points, 3), dtype=np.float32)
        
        return point_cloud.astype(np.float32)

    def predict_action_from_obs(self, obs_dict):
        """Predict action from observation dict (EXACT from eval_flowpolicy.py methodology)"""
        # Convert observation format to match flow policy expectations
        # The model expects obs with shape [batch, n_obs_steps, ...] that gets reshaped
        batch_obs = {}
        
        # Create agent_pos by concatenating ee_states, gripper_states, joint_states
        if 'ee_states' in obs_dict and 'gripper_states' in obs_dict and 'joint_states' in obs_dict:
            # Handle framestack dimensions properly - handle both framestack and single obs
            if obs_dict['ee_states'].ndim == 2:  # (n_frames, 16) framestack
                ee_frames = obs_dict['ee_states']  # (n_frames, 16)
                gripper_frames = obs_dict['gripper_states']  # (n_frames, 1)
                joint_frames = obs_dict['joint_states']  # (n_frames, 7)
                
                # Concatenate for each frame
                agent_pos_frames = []
                for i in range(ee_frames.shape[0]):
                    agent_pos = np.concatenate([ee_frames[i], gripper_frames[i], joint_frames[i]])
                    agent_pos_frames.append(agent_pos)
                agent_pos_stacked = np.stack(agent_pos_frames, axis=0)  # (n_frames, 24)
            else:
                ee_latest = obs_dict['ee_states']
                gripper_latest = obs_dict['gripper_states'] 
                joint_latest = obs_dict['joint_states']
                # Expected: [ee(16) + gripper(1) + joint(7)] = 24 dimensions
                agent_pos = np.concatenate([ee_latest, gripper_latest, joint_latest])
                # Create fake framestack for single observation
                agent_pos_stacked = np.repeat(agent_pos[None], self.dataset.n_obs_steps, axis=0)  # (n_obs_steps, 24)
            
            # Convert to the format the model expects: [batch, n_obs_steps, feature_dim]
            batch_obs['agent_pos'] = torch.from_numpy(agent_pos_stacked).float().unsqueeze(0).to(self.device)
        
        # Handle RGB images - resize to match dataset format (84x84)
        if 'eye_in_hand_rgb' in obs_dict:
            eye_rgb = obs_dict['eye_in_hand_rgb']
            if eye_rgb.ndim == 4:  # (n_frames, C, H, W) framestack
                eye_frames = []
                for i in range(eye_rgb.shape[0]):
                    frame = eye_rgb[i]  # (C, H, W)
                    frame_pil = Image.fromarray((frame.transpose(1,2,0) * 255).astype(np.uint8))
                    frame_resized = frame_pil.resize((84, 84))
                    frame_tensor = torch.from_numpy(np.array(frame_resized).transpose(2,0,1) / 255.0).float()
                    eye_frames.append(frame_tensor)
                eye_stacked = torch.stack(eye_frames, dim=0)  # (n_frames, C, H, W)
            else:
                # Single frame
                eye_rgb_pil = Image.fromarray((eye_rgb.transpose(1,2,0) * 255).astype(np.uint8))
                eye_rgb_resized = eye_rgb_pil.resize((84, 84))
                eye_rgb_tensor = torch.from_numpy(np.array(eye_rgb_resized).transpose(2,0,1) / 255.0).float()
                # Create fake framestack for single observation
                eye_stacked = eye_rgb_tensor.unsqueeze(0).repeat(self.dataset.n_obs_steps, 1, 1, 1)  # (n_obs_steps, C, H, W)
            
            batch_obs['handeye'] = eye_stacked.unsqueeze(0).to(self.device)  # (batch, n_obs_steps, C, H, W)
        
        if 'agentview' in obs_dict:
            agent_rgb = obs_dict['agentview']
            if agent_rgb.ndim == 4:  # (n_frames, C, H, W) framestack
                agent_frames = []
                for i in range(agent_rgb.shape[0]):
                    frame = agent_rgb[i]  # (C, H, W)
                    frame_pil = Image.fromarray((frame.transpose(1,2,0) * 255).astype(np.uint8))
                    frame_resized = frame_pil.resize((84, 84))
                    frame_tensor = torch.from_numpy(np.array(frame_resized).transpose(2,0,1) / 255.0).float()
                    agent_frames.append(frame_tensor)
                agent_stacked = torch.stack(agent_frames, dim=0)  # (n_frames, C, H, W)
            else:
                # Single frame  
                agent_rgb_pil = Image.fromarray((agent_rgb.transpose(1,2,0) * 255).astype(np.uint8))
                agent_rgb_resized = agent_rgb_pil.resize((84, 84))
                agent_rgb_tensor = torch.from_numpy(np.array(agent_rgb_resized).transpose(2,0,1) / 255.0).float()
                # Create fake framestack for single observation
                agent_stacked = agent_rgb_tensor.unsqueeze(0).repeat(self.dataset.n_obs_steps, 1, 1, 1)  # (n_obs_steps, C, H, W)
            
            batch_obs['agentview'] = agent_stacked.unsqueeze(0).to(self.device)  # (batch, n_obs_steps, C, H, W)
        
        # Create proper point cloud from rendered depth (REAL point clouds instead of dummy!)
        if 'eye_in_hand_depth' in obs_dict:
            # Use rendered depth to create point clouds for each frame
            if obs_dict['eye_in_hand_depth'].ndim == 3:  # (n_frames, H, W) framestack
                point_cloud_frames = []
                for i in range(obs_dict['eye_in_hand_depth'].shape[0]):
                    depth_frame = obs_dict['eye_in_hand_depth'][i]  # (H, W)
                    pc_frame = self._depth_to_pointcloud(depth_frame)
                    point_cloud_frames.append(pc_frame)
                point_cloud_stacked = np.stack(point_cloud_frames, axis=0)  # (n_frames, 1024, 3)
            else:
                # Single depth frame
                depth_frame = obs_dict['eye_in_hand_depth']  # (H, W)
                point_cloud = self._depth_to_pointcloud(depth_frame)
                # Create framestack for single observation
                point_cloud_stacked = np.repeat(point_cloud[None], self.dataset.n_obs_steps, axis=0)  # (n_obs_steps, 1024, 3)
        else:
            # Fallback to dummy point cloud if no depth available
            print("[FLOW WARNING] No depth data available, using dummy point cloud")
            point_cloud = self._generate_dummy_point_cloud()
            point_cloud_stacked = np.repeat(point_cloud[None], self.dataset.n_obs_steps, axis=0)  # (n_obs_steps, 1024, 3)
        
        batch_obs['point_cloud'] = torch.from_numpy(point_cloud_stacked).float().unsqueeze(0).to(self.device)  # (batch, n_obs_steps, 1024, 3)
        
        # Create depth if expected (placeholder matching dataset format)
        dummy_depth = np.ones((1, 84, 84), dtype=np.float32) * 0.5
        # Create framestack for depth
        depth_stacked = np.repeat(dummy_depth[None], self.dataset.n_obs_steps, axis=0)  # (n_obs_steps, 1, 84, 84)
        batch_obs['handeye_depth'] = torch.from_numpy(depth_stacked).unsqueeze(0).to(self.device)  # (batch, n_obs_steps, 1, 84, 84)
        
        print(f"[FLOW DEBUG] Prepared obs keys: {list(batch_obs.keys())}")
        print(f"[FLOW DEBUG] agent_pos shape: {batch_obs['agent_pos'].shape}")
        print(f"[FLOW DEBUG] handeye shape: {batch_obs['handeye'].shape}")
        print(f"[FLOW DEBUG] agentview shape: {batch_obs['agentview'].shape}")
        print(f"[FLOW DEBUG] point_cloud shape: {batch_obs['point_cloud'].shape}")
        print(f"[FLOW DEBUG] handeye_depth shape: {batch_obs['handeye_depth'].shape}")
        
        # Predict actions using EXACT method from eval_flowpolicy.py
        with torch.no_grad():
            try:
                predicted_actions = self.model.predict_action(batch_obs)
                predicted_actions = predicted_actions[0]  # Remove batch dimension
                print(f"[FLOW DEBUG] Raw prediction shape: {predicted_actions.shape}")
            except Exception as e:
                print(f"[FLOW ERROR] Model prediction failed: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        # Denormalize predictions (EXACT from eval_flowpolicy.py)
        predicted_actions_denorm = self._denormalize_actions(predicted_actions)
        
        # Compute predicted delta pose (EXACT from eval_flowpolicy.py)
        delta_pose, gripper_action = self.compute_delta_pose(predicted_actions_denorm)
        
        return {
            'action': predicted_actions_denorm.cpu().numpy(),
            'delta_pose': delta_pose.cpu().numpy(),
            'gripper': gripper_action.cpu().item()
        }
    
    def _generate_dummy_point_cloud(self):
        """Generate a dummy point cloud that matches training data format."""
        # Create a grid of points in a reasonable workspace volume
        # Similar to what would come from depth->pointcloud conversion
        np.random.seed(42)  # For reproducible results
        
        # Create points in a reasonable robot workspace
        x = np.random.uniform(-0.3, 0.3, 1024)  # 60cm workspace width
        y = np.random.uniform(-0.3, 0.3, 1024)  # 60cm workspace depth  
        z = np.random.uniform(0.0, 0.4, 1024)   # 40cm workspace height
        
        point_cloud = np.stack([x, y, z], axis=-1).astype(np.float32)
        return point_cloud

# =============================================================================
# Framestack for Flow Policy (based on diffusion policy pattern)
# =============================================================================
class FrameStackForFlow:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.obs_history = {}
    
    def reset(self, init_obs):
        self.obs_history = {}
        for k in init_obs:
            self.obs_history[k] = collections.deque([init_obs[k][None] for _ in range(self.num_frames)], maxlen=self.num_frames)
        return {k: np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history}
    
    def add_new_obs(self, new_obs):
        for k in new_obs:
            if k in self.obs_history: 
                self.obs_history[k].append(new_obs[k][None])
        return {k: np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history}

def parse_args():
    p = argparse.ArgumentParser(description="Flow Matching Policy rollout with EXACT rendering from reference code")
    # Environment parameters (EXACT from reference)
    p.add_argument("--env_model_path", type=str, default="./output_updated_scene",
                   help="Path to the environment Gaussian model")
    p.add_argument("--robot_links_dir", type=str, default="./robot",
                   help="Path to robot link Gaussian models")
    p.add_argument("--source_path", type=str, default="./objects_data_large_org",
                   help="Path to source data")
    p.add_argument("--img_w", type=int, default=320)
    p.add_argument("--img_h", type=int, default=240)
    p.add_argument("--fov_y", type=float, default=55.0)
    p.add_argument("--z_scene_offset", type=float, default=0.01)
    p.add_argument("--x_base_offset", type=float, default=-0.045)
    p.add_argument("--y_base_offset", type=float, default=0.010)
    p.add_argument("--z_base_offset", type=float, default=0.09)
    p.add_argument("--outset", type=float, default=DEF_OUTSET)
    p.add_argument("--camera_offset", type=float, default=DEF_CAMERA_OFFSET)
    p.add_argument("--front_marker_height", type=float, default=DEF_FRONT_MARKER_HEIGHT)
    p.add_argument("--camera_tilt", type=float, default=DEF_CAMERA_TILT_DEGREES)
    p.add_argument("--invert_hand_eye", action="store_true",
                   help="use cTg instead of gTc if you see rigid offset")
    
    # Demo initialization (EXACT from reference code)
    p.add_argument("--demo_file", type=str, 
                   default="/home/carl_lab/Riad/graspbility/GraspSplats/new_background_grasp_data_low_depth_with_delta_agent.hdf5",
                   help="HDF5 demo file for robot initialization")
    p.add_argument("--demo_index", type=int, default=25,
                   help="Demo index to load for robot initialization")
    
    # Flow Policy specific arguments
    p.add_argument("--flow_checkpoint", type=str, 
                   default="/home/carl_lab/Riad/graspbility/FlowPolicy/FlowPolicy/flowpolicy_checkpoints/best_flowpolicy_epoch_1.pth",
                   help="Path to the flow policy checkpoint")
    p.add_argument("--max_episode_steps", type=int, default=100,
                   help="Maximum number of policy steps per episode")
    p.add_argument("--pos_scale", type=float, default=1.0,
                   help="Position action scaling factor")
    p.add_argument("--rot_scale", type=float, default=1.0,
                   help="Rotation action scaling factor")
    p.add_argument("--dt", type=float, default=0.02,
                   help="Time step for visualization")
    
    # Recognition & FoV (if needed)
    p.add_argument("--recog_assets_dir", type=str, default="./recog_dual_assets")
    p.add_argument("--recog_ckpt_path", type=str, 
                   default="./checkpoints/new_bg_red_candy_both_view/after_train_500_epochs.ckpt")
    
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("[INFO] Creating EXACT Flow Matching rollout with reference environment...")
    
    # Create environment using EXACT same method as reference code
    env = FrankaGaussianEnv(
        env_model_path=args.env_model_path, 
        robot_links_dir=args.robot_links_dir,
        source_path=args.source_path, 
        img_w=args.img_w, 
        img_h=args.img_h,
        invert_hand_eye=args.invert_hand_eye,
        z_scene_offset=args.z_scene_offset, 
        z_base_offset=args.z_base_offset,
        x_base_offset=args.x_base_offset, 
        y_base_offset=args.y_base_offset,
        fov_y_deg=args.fov_y, 
        outset=args.outset, 
        camera_offset=args.camera_offset,
        front_marker_height=args.front_marker_height, 
        camera_tilt=args.camera_tilt
    )
    
    # Initialize from demo (EXACT same as reference code)
    print(f"[INFO] Loading demo {args.demo_index} from {args.demo_file}")
    init = load_demo_init(args.demo_file, args.demo_index)
    initialize_env(env, init)
    print("[ENV] Robot initialized to demo pose.")
    
    # Load Flow Policy (using EXACT evaluator from eval_flowpolicy.py)
    print("[FLOW] Loading Flow Policy Evaluator...")
    flow_policy = FlowPolicyEvaluator(
        checkpoint_path=args.flow_checkpoint,
        dataset_path=args.demo_file,
        device=device
    )
    
    # Setup framestack (simplified)
    n_obs_steps = 2  # Simple framestack
    framestacker = FrameStackForFlow(num_frames=n_obs_steps)
    print(f"[FLOW] Using {n_obs_steps} observation steps for FlowPolicy model.")
    
    # Recognition setup
    recognizer = DualViewRecognizer(
        assets_dir=args.recog_assets_dir,
        ckpt_path=args.recog_ckpt_path,
        device=device
    )
    
    # Initialize observation stack
    obs_dict = env._get_obs()
    obs_stack = framestacker.reset(obs_dict)
    
    # Setup visualization windows
    cv2.namedWindow("Hand-Eye View", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Agent View", cv2.WINDOW_NORMAL)
    
    print("[INFO] Starting Flow Matching rollout... Press 'q' to quit")
    
    # Policy rollout loop (EXACT structure from reference)
    stop = False
    for t in range(args.max_episode_steps):
        # Get policy prediction using Flow Matching
        try:
            action_result = flow_policy.predict_action_from_obs(obs_stack)
            delta_pose = action_result['delta_pose']
            gripper_action = action_result['gripper']
            
            # ========== DEBUG: FLOW POLICY ACTION ANALYSIS ==========
            print(f"\n🌊 [FLOW STEP {t+1}] ACTION DEBUG:")
            print(f"  Raw action shape: {action_result['action'].shape}")
            print(f"  Delta pose: {delta_pose}")
            print(f"    Position: {delta_pose[:3]}")
            print(f"    Rotation: {delta_pose[3:6]}")
            print(f"  Gripper: {gripper_action}")
            
            # Scale actions
            dp_tool = delta_pose[:3] * args.pos_scale
            rvec_tool = delta_pose[3:6] * args.rot_scale
            grip_width_m = float(gripper_action)
            
            print(f"  After scaling (pos_scale={args.pos_scale}, rot_scale={args.rot_scale}):")
            print(f"    Scaled position: {dp_tool}")
            print(f"    Scaled rotation: {rvec_tool}")
            print(f"    Gripper width: {grip_width_m}")
            print("🌊 =" * 50)
            
        except Exception as e:
            print(f"[ERROR] Flow policy prediction failed: {e}")
            break
        
        # Execute action using EXACT method from reference
        obs_dict, _ = env.step_delta_tool_axisangle(dp_tool, rvec_tool, grip_width_m)
        
        # --------- scoring at the visited state ----------
        # recognizability
        eye_pil = robust_to_pil(obs_dict['eye_in_hand_rgb'])
        agent_pil = robust_to_pil(obs_dict['agentview'])
        s_eye, s_agent = recognizer.score_views(eye_pil, agent_pil)
        
        # Display (EXACT from reference code)
        img_h_rgb = (obs_dict['eye_in_hand_rgb'].transpose(1,2,0)*255).astype(np.uint8)
        img_a_rgb = (obs_dict['agentview'].transpose(1,2,0)*255).astype(np.uint8)
        agent_bgr = cv2.cvtColor(img_a_rgb, cv2.COLOR_RGB2BGR)
        hand_bgr = cv2.cvtColor(img_h_rgb, cv2.COLOR_RGB2BGR)
        
        # Add score overlay
        lines = [
            f"Flow Matching Policy",
            f"Recog Eye : {s_eye:+.3f}",
            f"Recog Agnt: {s_agent:+.3f}",
            f"Step: {t+1}/{args.max_episode_steps}",
            f"Demo: {args.demo_index}"
        ]
        y = 30
        for text in lines:
            cv2.putText(agent_bgr, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            y += 25
            
        cv2.putText(hand_bgr, f"Hand-Eye (Flow Policy Demo {args.demo_index})", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Hand-Eye View", hand_bgr)
        cv2.imshow("Agent View", agent_bgr)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop = True
            break
        
        time.sleep(args.dt)
        if stop: 
            break
        
        # Update observation stack
        obs_stack = framestacker.add_new_obs(obs_dict)
    
    cv2.destroyAllWindows()
    env.close()
    print(f"[FLOW] Flow Matching rollout complete after {t+1} steps.")

if __name__ == "__main__":
    main()
