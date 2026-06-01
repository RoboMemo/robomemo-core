"""
BONES-SEED Dataset Loader
加载 SOMA Uniform/Proportional BVH 和 G1 CSV 数据，
转为项目统一的 npy 格式。

BVH → (frames, num_joints, 3) 3D joint positions (forward kinematics)
G1 CSV → (frames, 29) joint angles in radians

用法:
    loader = BonesSeedLoader("bones-seed")
    
    # 加载单个 BVH → 3D 关节位置
    positions = loader.load_bvh("soma_uniform/bvh/210531/jump_and_land_heavy_001__A001.bvh")
    
    # 加载 G1 CSV → 关节角度 (radians)
    g1_angles = loader.load_g1_csv("g1/csv/210531/jump_and_land_heavy_001__A001.csv")
    
    # 批量加载
    dataset = loader.load_batch(format="soma_uniform", category="Locomotion", limit=100)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import re


class BVHParser:
    """解析 BVH 文件，提取骨架结构和运动数据。"""
    
    # SOMA skeleton 中我们关心的关节（用于 retarget）
    # 映射到简化的关节索引
    SOMA_RETARGET_JOINTS = [
        "Root", "Hips", "Spine1", "Spine2", "Chest",
        "Neck1", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
        "RightLeg", "RightShin", "RightFoot", "RightToeBase",
    ]
    # 对应 24 joint 的简化 skeleton（与现有 mocap_data 兼容）
    SOMA_TO_SIMPLE_24 = {
        "Root": 0, "Hips": 1, "Spine1": 2, "Spine2": 3, "Chest": 4,
        "Neck1": 5, "Head": 6,
        "LeftShoulder": 7, "LeftArm": 8, "LeftForeArm": 9, "LeftHand": 10,
        "RightShoulder": 11, "RightArm": 12, "RightForeArm": 13, "RightHand": 14,
        "LeftLeg": 15, "LeftShin": 16, "LeftFoot": 17, "LeftToeBase": 18,
        "RightLeg": 19, "RightShin": 20, "RightFoot": 21, "RightToeBase": 22,
        # index 23 reserved for "center of mass" or additional
    }
    
    def __init__(self):
        self.joints = []       # list of joint names in channel order
        self.offsets = {}      # joint_name -> (3,) offset
        self.parents = {}      # joint_name -> parent_name
        self.channels = {}     # joint_name -> list of channel names
        self.channel_order = []  # flat list of (joint_name, channel_name)
        self.end_sites = set()
        
    def parse(self, filepath: str) -> Tuple[Dict, np.ndarray]:
        """Parse BVH file, return (skeleton_info, motion_data).
        
        Returns:
            skeleton: dict with joints, offsets, parents, channels
            frames: (num_frames, num_channels) raw channel data
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse hierarchy
        i = 0
        joint_stack = []
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("ROOT") or line.startswith("JOINT"):
                parts = line.split()
                joint_name = parts[1]
                self.joints.append(joint_name)
                
                if joint_stack:
                    self.parents[joint_name] = joint_stack[-1]
                else:
                    self.parents[joint_name] = None
                    
            elif line == "{":
                if self.joints:
                    joint_stack.append(self.joints[-1])
                    
            elif line == "}":
                if joint_stack:
                    joint_stack.pop()
                    
            elif line.startswith("OFFSET"):
                parts = line.split()
                offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                if self.joints:
                    current = joint_stack[-1] if joint_stack else self.joints[-1]
                    if current not in self.offsets:
                        self.offsets[current] = offset
                        
            elif line.startswith("CHANNELS"):
                parts = line.split()
                num_ch = int(parts[1])
                ch_names = parts[2:2+num_ch]
                current = joint_stack[-1] if joint_stack else self.joints[-1]
                self.channels[current] = ch_names
                for ch in ch_names:
                    self.channel_order.append((current, ch))
                    
            elif line.startswith("End Site"):
                # Skip end site
                pass
                
            elif line == "MOTION":
                i += 1
                break
                
            i += 1
        
        # Parse motion data
        num_frames = 0
        frame_time = 0.0
        frames_data = []
        
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Frames:"):
                num_frames = int(line.split(":")[1].strip())
            elif line.startswith("Frame Time:"):
                frame_time = float(line.split(":")[1].strip())
            elif line and not line.startswith("Frames") and not line.startswith("Frame"):
                values = [float(x) for x in line.split()]
                frames_data.append(values)
            i += 1
        
        motion = np.array(frames_data, dtype=np.float64)
        
        skeleton = {
            "joints": self.joints,
            "offsets": self.offsets,
            "parents": self.parents,
            "channels": self.channels,
            "channel_order": self.channel_order,
            "frame_time": frame_time,
            "num_frames": num_frames,
        }
        
        return skeleton, motion
    
    def forward_kinematics(self, skeleton: Dict, motion: np.ndarray) -> np.ndarray:
        """Compute 3D joint positions via forward kinematics.
        
        Args:
            skeleton: from parse()
            motion: (num_frames, num_channels)
            
        Returns:
            positions: (num_frames, num_joints, 3) in meters
        """
        joints = skeleton["joints"]
        num_frames = motion.shape[0]
        num_joints = len(joints)
        
        positions = np.zeros((num_frames, num_joints, 3))
        
        # Build channel index map: joint_name -> start_idx in motion row
        ch_idx = {}
        idx = 0
        for joint in joints:
            if joint in skeleton["channels"]:
                ch_idx[joint] = idx
                idx += len(skeleton["channels"][joint])
        
        for frame in range(num_frames):
            row = motion[frame]
            local_transforms = {}
            
            for j, joint in enumerate(joints):
                if joint not in skeleton["channels"]:
                    continue
                    
                channels = skeleton["channels"][joint]
                start = ch_idx[joint]
                
                # Extract translation and rotation
                tx, ty, tz = 0.0, 0.0, 0.0
                rx, ry, rz = 0.0, 0.0, 0.0
                
                for k, ch_name in enumerate(channels):
                    val = row[start + k]
                    if ch_name == "Xposition": tx = val
                    elif ch_name == "Yposition": ty = val
                    elif ch_name == "Zposition": tz = val
                    elif ch_name == "Xrotation": rx = val
                    elif ch_name == "Yrotation": ry = val
                    elif ch_name == "Zrotation": rz = val
                
                # Convert to radians
                rx, ry, rz = np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)
                
                # Rotation matrices (BVH uses ZYX order for SOMA)
                # The channel order tells us the rotation application order
                rot = self._euler_to_matrix(rx, ry, rz, channels)
                
                # Offset (convert cm to meters for SOMA which uses cm)
                offset = skeleton["offsets"].get(joint, np.zeros(3)) / 100.0
                
                # Translation from root
                trans = np.array([tx, ty, tz]) / 100.0
                
                # Build 4x4 transform
                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = offset + trans
                
                local_transforms[joint] = T
            
            # Chain transforms
            global_transforms = {}
            for j, joint in enumerate(joints):
                if joint not in local_transforms:
                    continue
                parent = skeleton["parents"].get(joint)
                if parent and parent in global_transforms:
                    global_transforms[joint] = global_transforms[parent] @ local_transforms[joint]
                else:
                    global_transforms[joint] = local_transforms[joint]
                
                positions[frame, j] = global_transforms[joint][:3, 3]
        
        return positions
    
    def _euler_to_matrix(self, rx, ry, rz, channels):
        """Convert Euler angles to rotation matrix respecting BVH channel order."""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Apply rotations in the order specified by channels
        rot_map = {"Xrotation": Rx, "Yrotation": Ry, "Zrotation": Rz}
        rot_channels = [ch for ch in channels if ch.endswith("rotation")]
        
        R = np.eye(3)
        for ch in rot_channels:
            R = R @ rot_map[ch]
        
        return R
    
    def extract_retarget_positions(self, skeleton, positions):
        """Extract the 24 key joint positions for retargeting.
        
        Args:
            skeleton: from parse()
            positions: (num_frames, num_all_joints, 3) from forward_kinematics()
            
        Returns:
            key_positions: (num_frames, 24, 3) compatible with existing mocap_data format
        """
        joints = skeleton["joints"]
        joint_idx = {name: i for i, name in enumerate(joints)}
        
        num_frames = positions.shape[0]
        key_pos = np.zeros((num_frames, 24, 3))
        
        for joint_name, target_idx in self.SOMA_TO_SIMPLE_24.items():
            if joint_name in joint_idx:
                key_pos[:, target_idx] = positions[:, joint_idx[joint_name]]
        
        # Index 23: center of mass approximation (average of hips)
        if "Hips" in joint_idx:
            key_pos[:, 23] = positions[:, joint_idx["Hips"]]
        
        return key_pos


class G1Loader:
    """加载 BONES-SEED G1 MuJoCo CSV 格式。"""
    
    # G1 CSV 中的关节列（去掉 Frame 和 root 6DOF）
    G1_JOINT_COLUMNS = [
        "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
        "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
        "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
        "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
        "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
        "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
        "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof",
        "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
        "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof",
        "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof", "right_wrist_yaw_joint_dof",
    ]
    
    # 映射到现有 sonic_retarget 的 G1 joint_map（23 joints）
    # sonic_retarget G1 indices:
    #   0=waist_yaw, 1=waist_roll, 2=waist_pitch,
    #   3=left_hip_yaw, 4=left_hip_roll, 5=left_hip_pitch,
    #   6=left_knee, 7=left_ankle_pitch, 8=left_ankle_roll,
    #   9=right_hip_yaw, 10=right_hip_roll, 11=right_hip_pitch,
    #   12=right_knee, 13=right_ankle_pitch, 14=right_ankle_roll,
    #   15=left_shoulder_pitch, 16=left_shoulder_roll, 17=left_elbow, 18=left_wrist,
    #   19=right_shoulder_pitch, 20=right_shoulder_roll, 21=right_elbow, 22=right_wrist
    G1_CSV_TO_SONIC = {
        "waist_yaw_joint_dof": 0,
        "waist_roll_joint_dof": 1,
        "waist_pitch_joint_dof": 2,
        "left_hip_yaw_joint_dof": 3,
        "left_hip_roll_joint_dof": 4,
        "left_hip_pitch_joint_dof": 5,
        "left_knee_joint_dof": 6,
        "left_ankle_pitch_joint_dof": 7,
        "left_ankle_roll_joint_dof": 8,
        "right_hip_yaw_joint_dof": 9,
        "right_hip_roll_joint_dof": 10,
        "right_hip_pitch_joint_dof": 11,
        "right_knee_joint_dof": 12,
        "right_ankle_pitch_joint_dof": 13,
        "right_ankle_roll_joint_dof": 14,
        "left_shoulder_pitch_joint_dof": 15,
        "left_shoulder_roll_joint_dof": 16,
        "left_elbow_joint_dof": 17,
        "left_wrist_roll_joint_dof": 18,  # map wrist_roll → wrist slot
        "right_shoulder_pitch_joint_dof": 19,
        "right_shoulder_roll_joint_dof": 20,
        "right_elbow_joint_dof": 21,
        "right_wrist_roll_joint_dof": 22,
    }
    
    @staticmethod
    def load_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load G1 CSV file.
        
        Returns:
            root_state: (frames, 7) - translateXYZ + rotateXYZ + frame
            joint_angles: (frames, 29) - all joint DOFs in degrees
        """
        df = pd.read_csv(filepath)
        
        root_cols = ["root_translateX", "root_translateY", "root_translateZ",
                     "root_rotateX", "root_rotateY", "root_rotateZ"]
        
        joint_cols = [c for c in df.columns if c.endswith("_dof")]
        
        root_state = df[root_cols].values
        joint_angles = df[joint_cols].values  # degrees
        
        return root_state, joint_angles
    
    @classmethod
    def to_sonic_format(cls, filepath: str) -> np.ndarray:
        """Load G1 CSV and convert to sonic_retarget compatible format.
        
        Returns:
            angles: (frames, 23) in radians, matching SONICRetargeter G1 joint_map
        """
        df = pd.read_csv(filepath)
        
        joint_cols = [c for c in df.columns if c.endswith("_dof")]
        num_frames = len(df)
        
        out = np.zeros((num_frames, 23))
        
        for col in joint_cols:
            if col in cls.G1_CSV_TO_SONIC:
                idx = cls.G1_CSV_TO_SONIC[col]
                out[:, idx] = np.deg2rad(df[col].values)
        
        return out


class BonesSeedLoader:
    """主加载器，集成 BVH 和 G1 数据。"""
    
    def __init__(self, dataset_root: str):
        self.root = Path(dataset_root)
        self._metadata = None
        self._bvh_parser_cache = {}
    
    @property
    def metadata(self) -> pd.DataFrame:
        """加载并缓存 metadata。"""
        if self._metadata is None:
            parquet_path = self.root / "metadata" / "seed_metadata_v003.parquet"
            csv_path = self.root / "metadata" / "seed_metadata_v003.csv"
            
            if parquet_path.exists():
                self._metadata = pd.read_parquet(parquet_path)
            elif csv_path.exists():
                self._metadata = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(f"No metadata found in {self.root / 'metadata'}")
        
        return self._metadata
    
    def list_motions(self, 
                     category: Optional[str] = None, 
                     package: Optional[str] = None,
                     movement_type: Optional[str] = None,
                     limit: Optional[int] = None,
                     mirror: bool = False) -> pd.DataFrame:
        """Filter and list available motions.
        
        Args:
            category: e.g., "Basic Locomotion Neutral", "Dancing"
            package: e.g., "Locomotion", "Communication"
            movement_type: e.g., "walking", "jogging"
            limit: max number of results
            mirror: include mirrored versions
        """
        df = self.metadata.copy()
        
        if not mirror:
            df = df[df["is_mirror"] == False]
        
        if category:
            df = df[df["category"].str.contains(category, case=False, na=False)]
        if package:
            df = df[df["package"].str.contains(package, case=False, na=False)]
        if movement_type:
            df = df[df["content_type_of_movement"].str.contains(movement_type, case=False, na=False)]
        
        if limit:
            df = df.head(limit)
        
        return df
    
    def load_bvh(self, relative_path: str, extract_key_joints: bool = True) -> np.ndarray:
        """Load a SOMA BVH file and compute 3D joint positions.
        
        Args:
            relative_path: e.g., "soma_uniform/bvh/210531/jump_and_land_heavy_001__A001.bvh"
            extract_key_joints: if True, return (frames, 24, 3) key joints only
            
        Returns:
            positions: (frames, 24, 3) or (frames, all_joints, 3)
        """
        filepath = self.root / relative_path
        
        parser = BVHParser()
        skeleton, motion = parser.parse(str(filepath))
        positions = parser.forward_kinematics(skeleton, motion)
        
        if extract_key_joints:
            return parser.extract_retarget_positions(skeleton, positions)
        
        return positions
    
    def load_g1_csv(self, relative_path: str, as_sonic: bool = True) -> np.ndarray:
        """Load a G1 CSV file.
        
        Args:
            relative_path: e.g., "g1/csv/210531/jump_and_land_heavy_001__A001.csv"
            as_sonic: if True, convert to SONICRetargeter G1 format (23 joints, radians)
            
        Returns:
            if as_sonic: (frames, 23) radians
            else: (frames, 29) degrees
        """
        filepath = self.root / relative_path
        
        if as_sonic:
            return G1Loader.to_sonic_format(str(filepath))
        else:
            _, angles = G1Loader.load_csv(str(filepath))
            return angles
    
    def load_motion_by_name(self, motion_name: str, format: str = "soma_uniform") -> dict:
        """Load a motion by its name from metadata.
        
        Args:
            motion_name: e.g., "jump_and_land_heavy_001__A001"
            format: "soma_uniform", "soma_proportional", or "g1"
            
        Returns:
            dict with 'positions' (BVH) or 'angles' (G1), plus metadata
        """
        row = self.metadata[self.metadata["filename"] == motion_name]
        if len(row) == 0:
            raise ValueError(f"Motion '{motion_name}' not found in metadata")
        
        row = row.iloc[0]
        result = {
            "name": motion_name,
            "category": row["category"],
            "package": row["package"],
            "duration_frames": row["move_duration_frames"],
            "description": row.get("content_natural_desc_1", ""),
        }
        
        if format == "g1":
            path = row["move_g1_mujoco_path"]
            result["angles"] = self.load_g1_csv(path, as_sonic=True)
            result["format"] = "g1_sonic"
        else:
            path_col = f"move_{format}_path"
            result["positions"] = self.load_bvh(row[path_col])
            result["format"] = format
        
        return result
    
    def load_batch(self, 
                   format: str = "soma_uniform",
                   category: Optional[str] = None,
                   package: Optional[str] = None,
                   limit: int = 100,
                   save_dir: Optional[str] = None) -> List[dict]:
        """Batch load motions, optionally save as npy.
        
        Args:
            format: "soma_uniform", "soma_proportional", "g1"
            category: filter by category
            package: filter by package
            limit: max motions to load
            save_dir: if provided, save each motion as {name}_{format}.npy
            
        Returns:
            list of dicts from load_motion_by_name
        """
        motions_df = self.list_motions(category=category, package=package, limit=limit)
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        for _, row in motions_df.iterrows():
            name = row["filename"]
            try:
                data = self.load_motion_by_name(name, format=format)
                results.append(data)
                
                if save_dir:
                    key = "angles" if format == "g1" else "positions"
                    np.save(Path(save_dir) / f"{name}_{format}.npy", data[key])
                    
            except Exception as e:
                print(f"⚠️  Failed to load {name}: {e}")
                continue
        
        return results
    
    def get_g1_ground_truth(self, motion_name: str) -> np.ndarray:
        """Get the official G1 retarget as ground truth for comparison.
        
        Returns:
            (frames, 23) in radians, SONICRetargeter G1 format
        """
        return self.load_motion_by_name(motion_name, format="g1")["angles"]
    
    def compare_retarget_quality(self, 
                                  motion_name: str, 
                                  our_retarget: np.ndarray) -> dict:
        """Compare our retarget output against BONES-SEED's G1 ground truth.
        
        Args:
            motion_name: motion to compare
            our_retarget: (frames, 23) our retarget in radians
            
        Returns:
            dict with MAE, RMSE, per-joint errors
        """
        gt = self.get_g1_ground_truth(motion_name)
        
        # Align frame counts
        min_frames = min(len(gt), len(our_retarget))
        gt = gt[:min_frames]
        ours = our_retarget[:min_frames]
        
        diff = gt - ours
        mae = np.abs(diff).mean(axis=0)  # per-joint MAE
        rmse = np.sqrt((diff ** 2).mean(axis=0))  # per-joint RMSE
        
        joint_names = [
            "waist_yaw", "waist_roll", "waist_pitch",
            "L_hip_yaw", "L_hip_roll", "L_hip_pitch",
            "L_knee", "L_ankle_pitch", "L_ankle_roll",
            "R_hip_yaw", "R_hip_roll", "R_hip_pitch",
            "R_knee", "R_ankle_pitch", "R_ankle_roll",
            "L_shoulder_pitch", "L_shoulder_roll", "L_elbow", "L_wrist",
            "R_shoulder_pitch", "R_shoulder_roll", "R_elbow", "R_wrist",
        ]
        
        return {
            "motion": motion_name,
            "num_frames": min_frames,
            "overall_mae_rad": float(np.abs(diff).mean()),
            "overall_mae_deg": float(np.rad2deg(np.abs(diff).mean())),
            "overall_rmse_rad": float(np.sqrt((diff ** 2).mean())),
            "overall_rmse_deg": float(np.rad2deg(np.sqrt((diff ** 2).mean()))),
            "per_joint_mae_deg": {
                name: float(np.rad2deg(mae[i])) for i, name in enumerate(joint_names)
            },
            "per_joint_rmse_deg": {
                name: float(np.rad2deg(rmse[i])) for i, name in enumerate(joint_names)
            },
        }


# === CLI / Quick test ===
if __name__ == "__main__":
    import sys
    import json
    
    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "bones-seed"
    loader = BonesSeedLoader(dataset_root)
    
    print(f"📊 BONES-SEED Dataset: {len(loader.metadata)} motions")
    print(f"   Packages: {loader.metadata['package'].value_counts().to_dict()}")
    print(f"   Categories: {loader.metadata['category'].nunique()}")
    
    # Test G1 loading
    walking = loader.list_motions(movement_type="walking", limit=3)
    print(f"\n🚶 Sample walking motions:")
    for _, row in walking.iterrows():
        name = row["filename"]
        g1_path = row["move_g1_mujoco_path"]
        angles = loader.load_g1_csv(g1_path)
        print(f"   {name}: {angles.shape}, range [{np.rad2deg(angles.min()):.1f}°, {np.rad2deg(angles.max()):.1f}°]")
    
    # Test BVH loading (just one to verify FK)
    if len(walking) > 0:
        name = walking.iloc[0]["filename"]
        bvh_path = walking.iloc[0]["move_soma_uniform_path"]
        print(f"\n🦴 Testing BVH FK on: {name}")
        try:
            positions = loader.load_bvh(bvh_path)
            print(f"   Positions: {positions.shape}")
            print(f"   Height range: [{positions[:,:,1].min():.3f}, {positions[:,:,1].max():.3f}] m")
        except Exception as e:
            print(f"   ⚠️  BVH FK error: {e}")
    
    print("\n✅ Loader test complete")
