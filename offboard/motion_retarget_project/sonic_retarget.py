"""
SONIC Latent Retargeting - 将人类动作映射到机器人关节空间
扩展版：支持 BONES-SEED BVH 源数据

原始功能：从 (frames, 24, 3) 关节位置 retarget 到 H1/G1/Fourier
新增功能：
  - 从 BONES-SEED SOMA Uniform BVH 加载并 retarget
  - 批量处理 BONES-SEED 数据集
  - 用 G1 ground truth 验证 retarget 质量
"""
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import sys
import json


class SONICRetargeter:
    def __init__(self):
        self.robot_configs = {
            "H1": {
                "num_joints": 19,
                "joint_map": {
                    "waist_yaw": 0, "waist_roll": 1, "waist_pitch": 2,
                    "left_hip_yaw": 3, "left_hip_roll": 4, "left_hip_pitch": 5,
                    "left_knee": 6, "left_ankle_pitch": 7, "left_ankle_roll": 8,
                    "right_hip_yaw": 9, "right_hip_roll": 10, "right_hip_pitch": 11,
                    "right_knee": 12, "right_ankle_pitch": 13, "right_ankle_roll": 14,
                    "left_shoulder_pitch": 15, "left_shoulder_roll": 16,
                    "right_shoulder_pitch": 17, "right_shoulder_roll": 18,
                }
            },
            "G1": {
                "num_joints": 23,
                "joint_map": {
                    "waist_yaw": 0, "waist_roll": 1, "waist_pitch": 2,
                    "left_hip_yaw": 3, "left_hip_roll": 4, "left_hip_pitch": 5,
                    "left_knee": 6, "left_ankle_pitch": 7, "left_ankle_roll": 8,
                    "right_hip_yaw": 9, "right_hip_roll": 10, "right_hip_pitch": 11,
                    "right_knee": 12, "right_ankle_pitch": 13, "right_ankle_roll": 14,
                    "left_shoulder_pitch": 15, "left_shoulder_roll": 16, "left_elbow": 17, "left_wrist": 18,
                    "right_shoulder_pitch": 19, "right_shoulder_roll": 20, "right_elbow": 21, "right_wrist": 22,
                }
            },
            "Fourier": {
                "num_joints": 21,
                "joint_map": {
                    "waist_yaw": 0, "waist_roll": 1, "waist_pitch": 2,
                    "left_hip_yaw": 3, "left_hip_roll": 4, "left_hip_pitch": 5,
                    "left_knee": 6, "left_ankle_pitch": 7, "left_ankle_roll": 8,
                    "right_hip_yaw": 9, "right_hip_roll": 10, "right_hip_pitch": 11,
                    "right_knee": 12, "right_ankle_pitch": 13, "right_ankle_roll": 14,
                    "left_shoulder_pitch": 15, "left_shoulder_roll": 16, "left_elbow": 17,
                    "right_shoulder_pitch": 18, "right_shoulder_roll": 19, "right_elbow": 20,
                }
            }
        }
    
    def _safe_angle(self, v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.arccos(np.clip(cos_angle, -1, 1))
    
    def retarget_motion(self, human_motion, robot_name):
        """原始 retarget：从 (frames, 24, 3) 关节位置到机器人关节角度。
        
        Joint indices (24-joint layout from existing mocap_data):
            0=Root, 1=Hips, 2=Spine1, 3=Spine2, 4=Chest
            5=Neck, 6=Head
            7=LShoulder, 8=LArm, 9=LForeArm, 10=LHand
            11=RShoulder, 12=RArm, 13=RForeArm, 14=RHand
            15=LLeg(Hip), 16=LShin(Knee), 17=LFoot, 18=LToe
            19=RLeg(Hip), 20=RShin(Knee), 21=RFoot, 22=RToe
            23=CoM
        """
        num_frames = human_motion.shape[0]
        cfg = self.robot_configs[robot_name]
        jm = cfg["joint_map"]
        out = np.zeros((num_frames, cfg["num_joints"]))
        
        for i in range(num_frames):
            p = human_motion[i]  # (24, 3)
            
            # --- Waist (Hips → Spine1) ---
            sv = p[2] - p[1]  # Spine1 - Hips
            n = np.linalg.norm(sv) + 1e-8
            out[i, jm["waist_yaw"]]   = np.arctan2(sv[0], sv[2]) * 0.3
            out[i, jm["waist_roll"]]  = np.arcsin(np.clip(sv[1] / n, -1, 1)) * 0.3
            out[i, jm["waist_pitch"]] = np.arctan2(-sv[2], sv[1]) * 0.3
            
            # --- Left leg (LLeg→LShin→LFoot) ---
            lh2k = p[16] - p[15]  # Hip→Knee
            lk2a = p[17] - p[16]  # Knee→Ankle
            n1 = np.linalg.norm(lh2k) + 1e-8
            out[i, jm["left_hip_yaw"]]   = np.arctan2(lh2k[0], n1) * 0.5
            out[i, jm["left_hip_roll"]]  = np.arcsin(np.clip(lh2k[1] / n1, -1, 1)) * 0.5
            out[i, jm["left_hip_pitch"]] = np.arctan2(lh2k[1], n1) * 0.5
            out[i, jm["left_knee"]]      = self._safe_angle(-lh2k, lk2a) * 0.5
            if "left_ankle_pitch" in jm:
                la2t = p[18] - p[17]  # Ankle→Toe
                na = np.linalg.norm(la2t) + 1e-8
                out[i, jm["left_ankle_pitch"]] = np.arctan2(la2t[1], na) * 0.3
            if "left_ankle_roll" in jm:
                out[i, jm["left_ankle_roll"]] = 0.0
            
            # --- Right leg (RLeg→RShin→RFoot) ---
            rh2k = p[20] - p[19]  # Hip→Knee
            rk2a = p[21] - p[20]  # Knee→Ankle
            n2 = np.linalg.norm(rh2k) + 1e-8
            out[i, jm["right_hip_yaw"]]   = np.arctan2(rh2k[0], n2) * 0.5
            out[i, jm["right_hip_roll"]]  = np.arcsin(np.clip(rh2k[1] / n2, -1, 1)) * 0.5
            out[i, jm["right_hip_pitch"]] = np.arctan2(rh2k[1], n2) * 0.5
            out[i, jm["right_knee"]]      = self._safe_angle(-rh2k, rk2a) * 0.5
            if "right_ankle_pitch" in jm:
                ra2t = p[22] - p[21]
                nra = np.linalg.norm(ra2t) + 1e-8
                out[i, jm["right_ankle_pitch"]] = np.arctan2(ra2t[1], nra) * 0.3
            if "right_ankle_roll" in jm:
                out[i, jm["right_ankle_roll"]] = 0.0
            
            # --- Left arm (LShoulder→LArm→LForeArm→LHand) ---
            ls2a = p[8] - p[7]    # Shoulder→Arm
            la2f = p[9] - p[8]    # Arm→ForeArm
            n3 = np.linalg.norm(ls2a) + 1e-8
            out[i, jm["left_shoulder_pitch"]] = np.arctan2(ls2a[1], n3) * 0.5
            out[i, jm["left_shoulder_roll"]]  = np.arcsin(np.clip(ls2a[0] / n3, -1, 1)) * 0.5
            if "left_elbow" in jm:
                out[i, jm["left_elbow"]] = self._safe_angle(-ls2a, la2f) * 0.5
            if "left_wrist" in jm:
                lf2h = p[10] - p[9]  # ForeArm→Hand
                out[i, jm["left_wrist"]] = self._safe_angle(-la2f, lf2h) * 0.3
            
            # --- Right arm ---
            rs2a = p[12] - p[11]
            ra2f = p[13] - p[12]
            n4 = np.linalg.norm(rs2a) + 1e-8
            out[i, jm["right_shoulder_pitch"]] = np.arctan2(rs2a[1], n4) * 0.5
            out[i, jm["right_shoulder_roll"]]  = np.arcsin(np.clip(rs2a[0] / n4, -1, 1)) * 0.5
            if "right_elbow" in jm:
                out[i, jm["right_elbow"]] = self._safe_angle(-rs2a, ra2f) * 0.5
            if "right_wrist" in jm:
                rf2h = p[14] - p[13]
                out[i, jm["right_wrist"]] = self._safe_angle(-ra2f, rf2h) * 0.3
        
        return out
    
    def retarget_from_bvh(self, bvh_path: str, robot_name: str, 
                          dataset_root: str = "bones-seed",
                          downsample: int = 4) -> np.ndarray:
        """从 BONES-SEED BVH 文件 retarget 到机器人。
        
        BVH @ 120fps → downsample to 30fps by default.
        
        Args:
            bvh_path: relative path within dataset, or absolute path
            robot_name: "H1", "G1", or "Fourier"
            dataset_root: BONES-SEED dataset root
            downsample: frame skip factor (120fps/4=30fps)
            
        Returns:
            (frames, num_joints) joint angles in radians
        """
        from bones_seed_loader import BonesSeedLoader
        
        loader = BonesSeedLoader(dataset_root)
        positions = loader.load_bvh(bvh_path)  # (frames, 24, 3)
        
        # Downsample from 120fps
        if downsample > 1:
            positions = positions[::downsample]
        
        return self.retarget_motion(positions, robot_name)
    
    def batch_retarget_bones_seed(self, 
                                   dataset_root: str = "bones-seed",
                                   robot_names: Optional[List[str]] = None,
                                   category: Optional[str] = None,
                                   package: Optional[str] = None,
                                   limit: int = 100,
                                   output_dir: str = "retargeted_motions",
                                   downsample: int = 4) -> Dict:
        """批量从 BONES-SEED 数据集 retarget。
        
        Args:
            dataset_root: BONES-SEED 数据集根目录
            robot_names: 目标机器人列表，默认 ["H1", "G1", "Fourier"]
            category: 过滤类别
            package: 过滤包
            limit: 最大处理数
            output_dir: 输出目录
            downsample: 帧率降采样因子
            
        Returns:
            dict with stats
        """
        from bones_seed_loader import BonesSeedLoader
        
        if robot_names is None:
            robot_names = ["H1", "G1", "Fourier"]
        
        loader = BonesSeedLoader(dataset_root)
        motions = loader.list_motions(category=category, package=package, limit=limit)
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        stats = {"processed": 0, "failed": 0, "robots": {r: 0 for r in robot_names}}
        
        for _, row in motions.iterrows():
            name = row["filename"]
            bvh_rel = row["move_soma_uniform_path"]
            
            try:
                positions = loader.load_bvh(bvh_rel)
                if downsample > 1:
                    positions = positions[::downsample]
                
                for robot in robot_names:
                    angles = self.retarget_motion(positions, robot)
                    np.save(out_path / f"{name}_{robot}.npy", angles)
                    stats["robots"][robot] += 1
                
                stats["processed"] += 1
                
                if stats["processed"] % 50 == 0:
                    print(f"  ⏳ Processed {stats['processed']}/{len(motions)}...")
                    
            except Exception as e:
                print(f"  ⚠️  Failed {name}: {e}")
                stats["failed"] += 1
        
        return stats
    
    def validate_with_g1_ground_truth(self, 
                                       dataset_root: str = "bones-seed",
                                       motion_names: Optional[List[str]] = None,
                                       limit: int = 10) -> List[dict]:
        """用 BONES-SEED 自带的 G1 数据验证 retarget 质量。
        
        Pipeline:
            1. 加载 SOMA Uniform BVH → FK → 24-joint positions
            2. 用 SONICRetargeter.retarget_motion → G1 23-joint angles
            3. 加载 BONES-SEED G1 CSV ground truth
            4. 比较差异
            
        Returns:
            list of comparison dicts
        """
        from bones_seed_loader import BonesSeedLoader
        
        loader = BonesSeedLoader(dataset_root)
        
        if motion_names is None:
            # Pick some walking motions for validation
            df = loader.list_motions(movement_type="walking", limit=limit)
            motion_names = df["filename"].tolist()
        
        results = []
        for name in motion_names:
            try:
                # Our pipeline: BVH → FK → retarget
                data = loader.load_motion_by_name(name, format="soma_uniform")
                positions = data["positions"]
                positions = positions[::4]  # 120fps → 30fps
                our_g1 = self.retarget_motion(positions, "G1")
                
                # Ground truth
                comparison = loader.compare_retarget_quality(name, our_g1)
                results.append(comparison)
                
                print(f"  {name}: MAE={comparison['overall_mae_deg']:.2f}°, "
                      f"RMSE={comparison['overall_rmse_deg']:.2f}°")
                
            except Exception as e:
                print(f"  ⚠️  {name}: {e}")
        
        if results:
            avg_mae = np.mean([r["overall_mae_deg"] for r in results])
            avg_rmse = np.mean([r["overall_rmse_deg"] for r in results])
            print(f"\n📊 Average across {len(results)} motions: MAE={avg_mae:.2f}°, RMSE={avg_rmse:.2f}°")
        
        return results


# ============================================================
# CLI Entry Point
# ============================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SONIC Retarget with BONES-SEED support")
    sub = parser.add_subparsers(dest="command")
    
    # Original: retarget from existing mocap npy
    p_legacy = sub.add_parser("legacy", help="Retarget from existing mocap_data/*.npy")
    p_legacy.add_argument("--mocap-dir", default="mocap_data")
    p_legacy.add_argument("--output-dir", default="retargeted_motions")
    
    # New: retarget from BONES-SEED
    p_bones = sub.add_parser("bones", help="Batch retarget from BONES-SEED dataset")
    p_bones.add_argument("--dataset", default="bones-seed", help="BONES-SEED root dir")
    p_bones.add_argument("--robots", nargs="+", default=["H1", "G1", "Fourier"])
    p_bones.add_argument("--category", default=None)
    p_bones.add_argument("--package", default=None)
    p_bones.add_argument("--limit", type=int, default=100)
    p_bones.add_argument("--output-dir", default="retargeted_motions_bones")
    p_bones.add_argument("--downsample", type=int, default=4, help="Frame skip (120fps/N)")
    
    # Validate against G1 ground truth
    p_val = sub.add_parser("validate", help="Validate retarget quality vs G1 ground truth")
    p_val.add_argument("--dataset", default="bones-seed")
    p_val.add_argument("--limit", type=int, default=10)
    p_val.add_argument("--output", default=None, help="Save results JSON to file")
    
    args = parser.parse_args()
    
    retargeter = SONICRetargeter()
    
    if args.command == "legacy" or args.command is None:
        # Original behavior
        mocap_dir = Path(args.mocap_dir if hasattr(args, 'mocap_dir') else "mocap_data")
        output_dir = Path(args.output_dir if hasattr(args, 'output_dir') else "retargeted_motions")
        output_dir.mkdir(exist_ok=True)
        
        for f in mocap_dir.glob("*.npy"):
            motion = np.load(f)
            for robot in ["H1", "G1", "Fourier"]:
                out = retargeter.retarget_motion(motion, robot)
                np.save(output_dir / f"{f.stem}_{robot}.npy", out)
                print(f"✅ {f.stem} → {robot}: {out.shape}, "
                      f"range [{out.min():.3f}, {out.max():.3f}] rad")
        
        print("\n✅ All retargeted motions saved")
    
    elif args.command == "bones":
        print(f"🦴 BONES-SEED Batch Retarget")
        print(f"   Dataset: {args.dataset}")
        print(f"   Robots: {args.robots}")
        print(f"   Output: {args.output_dir}")
        
        stats = retargeter.batch_retarget_bones_seed(
            dataset_root=args.dataset,
            robot_names=args.robots,
            category=args.category,
            package=args.package,
            limit=args.limit,
            output_dir=args.output_dir,
            downsample=args.downsample,
        )
        
        print(f"\n📊 Results:")
        print(f"   Processed: {stats['processed']}")
        print(f"   Failed: {stats['failed']}")
        for robot, count in stats['robots'].items():
            print(f"   {robot}: {count} files")
    
    elif args.command == "validate":
        print(f"🔍 Validating retarget quality against G1 ground truth...")
        results = retargeter.validate_with_g1_ground_truth(
            dataset_root=args.dataset,
            limit=args.limit,
        )
        
        if args.output and results:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
