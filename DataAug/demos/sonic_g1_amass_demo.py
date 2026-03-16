"""
SONIC × G1 × AMASS Demo
========================
End-to-end demo: AMASS motion capture → SONIC ONNX retarget → G1 joint targets → 3D visualization.

Supports two data modes:
  1. --retarget path/to/retargeted.npz  (unimotion/AMASS format: joint_pos(N,29) + body_pos_w(N,30,3))
  2. --synthetic walk|wave|squat|dance  (generates SMPL-like motions for testing)

Pipeline:
  Left panel:  Ground-truth body positions (from mocap/retarget data or SMPL FK)
  Right panel: G1 skeleton driven by SONIC ONNX decoder output

Usage:
    # Real AMASS data (recommended)
    python -m cross_embodiment_retarget_demo.demos.sonic_g1_amass_demo \
        --retarget data/amass/CMU_07_01_walk.npz --save-gif

    # Synthetic fallback
    python -m cross_embodiment_retarget_demo.demos.sonic_g1_amass_demo \
        --synthetic --motion walk --duration 5 --save-gif
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── G1 Body Names (30 bodies from IsaacLab) ──────────────────

G1_BODY_NAMES = [
    "pelvis",           # 0
    "left_hip_pitch",   # 1
    "right_hip_pitch",  # 2
    "waist_yaw",        # 3 (=pelvis, same position)
    "left_hip_roll",    # 4
    "right_hip_roll",   # 5
    "waist_roll",       # 6
    "left_hip_yaw",     # 7
    "right_hip_yaw",    # 8
    "waist_pitch",      # 9 (=torso_link)
    "left_knee",        # 10
    "right_knee",       # 11
    "left_shoulder_pitch", # 12
    "right_shoulder_pitch", # 13
    "left_ankle_pitch", # 14
    "right_ankle_pitch", # 15
    "left_shoulder_roll", # 16
    "right_shoulder_roll", # 17
    "left_ankle_roll",  # 18
    "right_ankle_roll",  # 19
    "left_shoulder_yaw", # 20
    "right_shoulder_yaw", # 21
    "left_elbow",       # 22
    "right_elbow",      # 23
    "left_wrist_roll",  # 24
    "right_wrist_roll",  # 25
    "left_wrist_pitch",  # 26
    "right_wrist_pitch", # 27
    "left_wrist_yaw",   # 28
    "right_wrist_yaw",  # 29
]

# Bones connecting G1 body nodes for skeleton visualization
G1_BODY_BONES = [
    # Spine
    (0, 3), (3, 6), (6, 9),
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10), (10, 14), (14, 18),
    # Right leg
    (0, 2), (2, 5), (5, 8), (8, 11), (11, 15), (15, 19),
    # Left arm
    (9, 12), (12, 16), (16, 20), (20, 22), (22, 24), (24, 26), (26, 28),
    # Right arm
    (9, 13), (13, 17), (17, 21), (21, 23), (23, 25), (25, 27), (27, 29),
]


# ── G1 Joint Names (29 DOF) ──────────────────────────────────

G1_JOINT_NAMES = [
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch",               # 0-2
    "l_knee", "l_ankle_pitch", "l_ankle_roll",               # 3-5
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch",               # 6-8
    "r_knee", "r_ankle_pitch", "r_ankle_roll",               # 9-11
    "waist_yaw", "waist_roll", "waist_pitch",                # 12-14
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", # 15-17
    "l_elbow", "l_wrist_roll", "l_wrist_pitch", "l_wrist_yaw", # 18-21
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", # 22-24
    "r_elbow", "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw", # 25-28
]


# ── Simplified FK for joint-angle visualization ───────────────

def joint_angles_to_keypoints(joint_angles: np.ndarray) -> dict[str, np.ndarray]:
    """Convert 29 G1 joint angles to approximate 3D keypoints via trig FK.

    Uses 2-link planar FK in the sagittal (YZ) plane.

    The joint ordering here follows the unimotion/AMASS retarget format:
      0-5:   left leg  (hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll)
      6-11:  right leg
      12-14: waist (yaw, roll, pitch)
      15-21: left arm  (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
      22-28: right arm
    """
    q = joint_angles
    kp = {}

    # Pelvis
    pelvis = np.array([0.0, 0.0, 0.75])
    kp["pelvis"] = pelvis

    # Torso chain (waist: q12=yaw, q13=roll, q14=pitch)
    torso = pelvis + np.array([
        np.sin(q[12]) * 0.04,
        np.sin(q[14]) * 0.04,
        0.30,
    ])
    kp["torso"] = torso

    # Helper: 2-link FK in YZ plane. angle=0 → pointing down (-Z)
    def fk_yz(origin, a1, L1, a2, L2):
        j1 = origin + np.array([0, L1 * np.sin(a1), -L1 * np.cos(a1)])
        a_tot = a1 + a2
        j2 = j1 + np.array([0, L2 * np.sin(a_tot), -L2 * np.cos(a_tot)])
        return j1, j2

    # ── Left Leg (q0-q5) ──
    l_hip = pelvis + np.array([0.09, 0, 0])
    kp["l_hip"] = l_hip
    thigh, shin = 0.35, 0.37
    l_knee, l_ankle = fk_yz(l_hip, q[2], thigh, q[3], shin)
    l_knee[0] += np.sin(q[0]) * 0.05  # hip yaw
    l_ankle[0] += np.sin(q[0]) * 0.08
    kp["l_knee"] = l_knee
    kp["l_ankle"] = l_ankle
    kp["l_foot"] = np.array([l_ankle[0], l_ankle[1] + 0.08, max(0.0, l_ankle[2] - 0.05)])

    # ── Right Leg (q6-q11) ──
    r_hip = pelvis + np.array([-0.09, 0, 0])
    kp["r_hip"] = r_hip
    r_knee, r_ankle = fk_yz(r_hip, q[8], thigh, q[9], shin)
    r_knee[0] -= np.sin(q[6]) * 0.05
    r_ankle[0] -= np.sin(q[6]) * 0.08
    kp["r_knee"] = r_knee
    kp["r_ankle"] = r_ankle
    kp["r_foot"] = np.array([r_ankle[0], r_ankle[1] + 0.08, max(0.0, r_ankle[2] - 0.05)])

    # ── Left Arm (q15-q21) ──
    l_sh = torso + np.array([0.18, 0, 0.05])
    kp["l_shoulder"] = l_sh
    upper_arm, forearm = 0.26, 0.24
    l_elbow, l_wrist = fk_yz(l_sh, q[15], upper_arm, q[18], forearm)
    l_elbow[0] += np.sin(q[17]) * 0.05
    l_wrist[0] += np.sin(q[17]) * 0.08
    kp["l_elbow"] = l_elbow
    kp["l_wrist"] = l_wrist
    kp["l_hand"] = l_wrist + np.array([0, 0, -0.05])

    # ── Right Arm (q22-q28) ──
    r_sh = torso + np.array([-0.18, 0, 0.05])
    kp["r_shoulder"] = r_sh
    r_elbow, r_wrist = fk_yz(r_sh, q[22], upper_arm, q[25], forearm)
    r_elbow[0] -= np.sin(q[24]) * 0.05
    r_wrist[0] -= np.sin(q[24]) * 0.08
    kp["r_elbow"] = r_elbow
    kp["r_wrist"] = r_wrist
    kp["r_hand"] = r_wrist + np.array([0, 0, -0.05])

    # Neck / Head
    kp["neck"] = torso + np.array([0, 0, 0.10])

    return kp


# Simplified skeleton bones for the joint-angle visualization
G1_SIMPLE_BONES = [
    ("pelvis", "torso"), ("torso", "neck"),
    ("torso", "l_shoulder"), ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"), ("l_wrist", "l_hand"),
    ("torso", "r_shoulder"), ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"), ("r_wrist", "r_hand"),
    ("pelvis", "l_hip"), ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"), ("l_ankle", "l_foot"),
    ("pelvis", "r_hip"), ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"), ("r_ankle", "r_foot"),
]


# ── SMPL Bones (for synthetic mode) ──────────────────────────

SMPL_BONES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (20, 22), (21, 23),
]


# ── Main Demo ─────────────────────────────────────────────────

def run_demo(args: argparse.Namespace):
    """Run the SONIC × G1 × AMASS end-to-end demo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime required. Install: pip install onnxruntime-gpu")
        sys.exit(1)

    print("=" * 60)
    print("  SONIC × G1 × AMASS  End-to-End Demo")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────
    use_retarget = args.retarget is not None
    motion = None
    retarget_data = None

    if use_retarget:
        # Load retargeted AMASS data (unimotion format)
        retarget_path = Path(args.retarget)
        if not retarget_path.exists():
            print(f"❌ File not found: {retarget_path}")
            sys.exit(1)

        print(f"\n📂 Loading retargeted AMASS: {retarget_path}")
        retarget_data = np.load(str(retarget_path), allow_pickle=True)

        fps = int(retarget_data["fps"][0])
        gt_joint_pos = retarget_data["joint_pos"]     # (N, 29) G1 joint angles
        gt_body_pos = retarget_data["body_pos_w"]      # (N, 30, 3) body world positions
        gt_joint_vel = retarget_data["joint_vel"]       # (N, 29)

        n_total = gt_joint_pos.shape[0]
        print(f"   Frames: {n_total}, FPS: {fps}, Duration: {n_total/fps:.1f}s")
        print(f"   Joint range: [{gt_joint_pos.min():.3f}, {gt_joint_pos.max():.3f}]")

    else:
        # Synthetic mode
        from ..src.amass_loader import SyntheticAMASS
        print(f"\n🎭 Generating synthetic motion: {args.motion}, {args.duration}s")
        synth = SyntheticAMASS(fps=args.fps)
        motion = synth.generate(motion_type=args.motion, duration=args.duration)
        fps = int(motion.fps)
        n_total = motion.n_frames
        print(f"   Frames: {n_total}, FPS: {fps}, Duration: {motion.duration:.1f}s")

    # ── Load SONIC ONNX models ────────────────────────────────
    model_dir = Path(args.model_dir).expanduser()
    encoder_path = model_dir / "model_encoder.onnx"
    decoder_path = model_dir / "model_decoder.onnx"

    if not encoder_path.exists() or not decoder_path.exists():
        print(f"\n❌ SONIC models not found in {model_dir}")
        sys.exit(1)

    device = args.device
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device == "cuda" else ["CPUExecutionProvider"])

    print(f"\n🧠 Loading SONIC ONNX models ({device})...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    encoder = ort.InferenceSession(str(encoder_path), sess_options=sess_opts, providers=providers)
    decoder = ort.InferenceSession(str(decoder_path), sess_options=sess_opts, providers=providers)

    enc_prov = encoder.get_providers()[0]
    dec_prov = decoder.get_providers()[0]
    print(f"   Encoder: {enc_prov}, Decoder: {dec_prov}")

    # ── Import bridge ─────────────────────────────────────────
    from ..src.sonic_amass_bridge import (
        SonicAMASSBridge, G1_DEFAULT_DOF_POS, G1_ACTION_SCALE,
        ENCODER_DIM, DECODER_DIM, NUM_G1_JOINTS,
    )

    bridge = SonicAMASSBridge(dt=1.0 / fps)

    n_frames = n_total
    if args.max_frames and args.max_frames < n_frames:
        n_frames = args.max_frames

    print(f"\n🚀 Running SONIC inference on {n_frames} frames...")

    # Storage
    gt_bodies_all = []        # left panel: ground truth body positions
    g1_joint_targets_all = [] # right panel: SONIC output joint angles
    encoder_tokens_all = []
    timing_stats = []

    for frame_idx in range(n_frames):
        t0 = time.perf_counter()

        if use_retarget:
            # ── Pack encoder from retargeted G1 joint data ──
            # In G1 mode, encoder needs reference G1 joint positions/velocities
            ref_joints = gt_joint_pos[frame_idx]  # (29,)
            ref_vel = gt_joint_vel[frame_idx]      # (29,)
            ref_body = gt_body_pos[frame_idx]      # (30, 3)

            # Root Z from pelvis body position
            root_z = float(ref_body[0, 2])

            # Root orientation as 6D (take from body quaternion)
            root_quat = retarget_data["body_quat_w"][frame_idx, 0]  # (4,)
            # Convert quaternion (w,x,y,z) to rotation matrix
            w, x, y, z = root_quat
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
            ], dtype=np.float32)
            anchor_6d = R[:, :2].T.flatten()  # (6,)

            # Pack encoder observation (G1 mode)
            bridge._motion_joint_pos_buf.push(ref_joints)
            bridge._motion_joint_vel_buf.push(np.clip(ref_vel, -10, 10))
            bridge._motion_root_z_buf.push(np.array([root_z], dtype=np.float32))
            bridge._motion_anchor_orient_buf.push(anchor_6d)

            # Lower body (indices 0-11 in this joint ordering = legs)
            lower_pos = ref_joints[:12]
            lower_vel = np.clip(ref_vel[:12], -10, 10)
            bridge._motion_lower_pos_buf.push(lower_pos)
            bridge._motion_lower_vel_buf.push(lower_vel)

            # Build encoder vector
            obs = np.zeros(ENCODER_DIM, dtype=np.float32)
            offset = 0

            # [0:4] mode = G1
            obs[0] = 1.0
            offset = 4

            # [4:294] joint_pos 10f step5
            obs[offset:offset+290] = bridge._motion_joint_pos_buf.get_contiguous(10, step=5)
            offset += 290

            # [294:584] joint_vel 10f step5
            obs[offset:offset+290] = bridge._motion_joint_vel_buf.get_contiguous(10, step=5)
            offset += 290

            # [584:594] root_z 10f step5
            obs[offset:offset+10] = bridge._motion_root_z_buf.get_contiguous(10, step=5)
            offset += 10

            # [594:595] root_z current
            obs[offset] = root_z
            offset += 1

            # [595:601] anchor orient current
            obs[offset:offset+6] = anchor_6d
            offset += 6

            # [601:661] anchor orient 10f step5
            obs[offset:offset+60] = bridge._motion_anchor_orient_buf.get_contiguous(10, step=5)
            offset += 60

            # [661:781] lower body pos 10f step5
            obs[offset:offset+120] = bridge._motion_lower_pos_buf.get_contiguous(10, step=5)
            offset += 120

            # [781:901] lower body vel 10f step5
            obs[offset:offset+120] = bridge._motion_lower_vel_buf.get_contiguous(10, step=5)
            offset += 120

            # [901:1762] — VR/SMPL/wrist targets = zeros for G1 mode
            enc_obs = obs.reshape(1, ENCODER_DIM)

            # Store ground truth body for left panel
            gt_bodies_all.append(ref_body.copy())
        else:
            # Synthetic mode: use SMPL bridge
            enc_obs = bridge.pack_encoder_input(motion, frame_idx)
            gt_bodies_all.append(motion.joint_positions[frame_idx].copy())

        t1 = time.perf_counter()

        # Run encoder
        tokens = encoder.run(None, {"obs_dict": enc_obs})[0]
        t2 = time.perf_counter()

        # Pack decoder
        dec_obs = bridge.pack_decoder_input(tokens)
        t3 = time.perf_counter()

        # Run decoder
        action = decoder.run(None, {"obs_dict": dec_obs})[0].flatten()
        t4 = time.perf_counter()

        # Apply action → joint targets
        action = np.clip(action, -5.0, 5.0)
        bridge.step_mock_physics(action, velocity_limit=args.vel_limit)
        t5 = time.perf_counter()

        g1_joint_targets_all.append(bridge.joint_positions.copy())
        encoder_tokens_all.append(tokens.flatten().copy())

        timing_stats.append({
            "pack": (t1 - t0) * 1000,
            "enc": (t2 - t1) * 1000,
            "dec": (t4 - t3) * 1000,
            "total": (t5 - t0) * 1000,
        })

        if frame_idx % 50 == 0 or frame_idx == n_frames - 1:
            s = timing_stats[-1]
            print(f"   Frame {frame_idx:4d}/{n_frames}: "
                  f"enc={s['enc']:.2f}ms  dec={s['dec']:.2f}ms  "
                  f"total={s['total']:.2f}ms  "
                  f"action=[{action.min():.2f}, {action.max():.2f}]")

    # ── Statistics ────────────────────────────────────────────
    print("\n" + "=" * 60)
    warmup = min(10, len(timing_stats) // 2)
    stats = {k: np.array([s[k] for s in timing_stats[warmup:]]) for k in timing_stats[0]}
    for key, vals in stats.items():
        print(f"   {key:8s}: mean={vals.mean():.3f}ms  p99={np.percentile(vals, 99):.3f}ms")
    print(f"   FPS: {1000.0 / stats['total'].mean():.1f} (target: {fps})")

    g1_targets = np.array(g1_joint_targets_all)
    tokens_arr = np.array(encoder_tokens_all)

    print(f"\n   G1 targets range: [{g1_targets.min():.4f}, {g1_targets.max():.4f}]")
    active = np.where(g1_targets.std(axis=0) > 0.01)[0]
    print(f"   Active joints: {len(active)}/{NUM_G1_JOINTS}")
    for j in active:
        name = G1_JOINT_NAMES[j] if j < len(G1_JOINT_NAMES) else f"joint_{j}"
        std = g1_targets[:, j].std()
        rng = [g1_targets[:, j].min(), g1_targets[:, j].max()]
        print(f"     {j:2d} {name:24s}: std={std:.4f}  range=[{rng[0]:.3f}, {rng[1]:.3f}]")

    if use_retarget:
        # Compare with ground truth
        gt_joints = gt_joint_pos[:n_frames]
        diff = g1_targets - gt_joints
        print(f"\n   Tracking error (vs GT): mean={np.abs(diff).mean():.4f} rad, "
              f"max={np.abs(diff).max():.4f} rad")

    # ── Visualization ─────────────────────────────────────────
    print(f"\n🎬 Generating animation ({n_frames} frames)...")

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("SONIC × G1 × AMASS Demo", fontsize=14, fontweight="bold")

    ax_gt = fig.add_subplot(121, projection="3d")
    ax_g1 = fig.add_subplot(122, projection="3d")

    # Compute centering for ground truth
    gt_bodies = np.array(gt_bodies_all)

    if use_retarget:
        # Center on pelvis (body 0) XY, keep Z absolute
        center_per_frame = gt_bodies[:, 0, :].copy()
        center_per_frame[:, 2] = 0  # don't shift Z
    else:
        center_per_frame = gt_bodies[:, 0, :].copy()
        center_per_frame[:, 2] = 0

    def set_axes(ax, x=(-0.6, 0.6), y=(-0.6, 0.6), z=(-0.1, 1.6)):
        ax.set_xlim(*x)
        ax.set_ylim(*y)
        ax.set_zlim(*z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=15, azim=-60)

    # Subsample for animation
    anim_step = max(1, n_frames // args.max_anim_frames)
    anim_indices = list(range(0, n_frames, anim_step))

    def update(anim_idx):
        frame = anim_indices[anim_idx]
        ax_gt.cla()
        ax_g1.cla()

        # ── Left: Ground truth ──
        if use_retarget:
            bodies = gt_bodies[frame] - center_per_frame[frame]
            ax_gt.set_title(f"Ground Truth (frame {frame})")
            set_axes(ax_gt)

            for (b1, b2) in G1_BODY_BONES:
                if b1 < len(bodies) and b2 < len(bodies):
                    p1, p2 = bodies[b1], bodies[b2]
                    ax_gt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                               "b-", linewidth=2, alpha=0.8)
            ax_gt.scatter(bodies[:, 0], bodies[:, 1], bodies[:, 2],
                          c="blue", s=15, alpha=0.8)
        else:
            # SMPL skeleton
            smpl_joints = gt_bodies[frame] - center_per_frame[frame]
            ax_gt.set_title(f"SMPL Input (frame {frame})")
            set_axes(ax_gt)

            for (j1, j2) in SMPL_BONES:
                p1, p2 = smpl_joints[j1], smpl_joints[j2]
                ax_gt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                           "b-", linewidth=2, alpha=0.8)
            ax_gt.scatter(smpl_joints[:, 0], smpl_joints[:, 1], smpl_joints[:, 2],
                          c="blue", s=15, alpha=0.8)

        # ── Right: G1 SONIC output ──
        kp = joint_angles_to_keypoints(g1_joint_targets_all[frame])
        ax_g1.set_title(f"G1 SONIC Output (frame {frame})")
        set_axes(ax_g1)

        for (n1, n2) in G1_SIMPLE_BONES:
            if n1 in kp and n2 in kp:
                p1, p2 = kp[n1], kp[n2]
                ax_g1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                           "r-", linewidth=2.5, alpha=0.8)
        pts = np.array([kp[n] for n in kp])
        ax_g1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="red", s=20, alpha=0.9)

        return []

    anim = FuncAnimation(fig, update, frames=len(anim_indices),
                         interval=1000.0 / min(fps, 30), blit=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_name = Path(args.retarget).stem if use_retarget else f"synthetic_{args.motion}"

    if args.save_gif:
        gif_path = output_dir / f"sonic_g1_{source_name}.gif"
        print(f"   Saving GIF: {gif_path}")
        writer = PillowWriter(fps=min(fps, 25))
        anim.save(str(gif_path), writer=writer, dpi=80)
        print(f"   ✅ Saved: {gif_path} ({gif_path.stat().st_size / 1024:.0f} KB)")

    # Joint trajectory plot
    traj_path = output_dir / f"sonic_g1_trajectories_{source_name}.png"
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig2.suptitle(f"G1 Joint Trajectories — {source_name}", fontsize=13)

    time_axis = np.arange(g1_targets.shape[0]) / fps

    # Waist joints
    for j in [12, 13, 14]:
        if j < len(G1_JOINT_NAMES):
            axes2[0].plot(time_axis, g1_targets[:, j], label=G1_JOINT_NAMES[j])
    if use_retarget:
        for j in [12, 13, 14]:
            axes2[0].plot(time_axis, gt_joint_pos[:n_frames, j], "--",
                          alpha=0.5, label=f"GT_{G1_JOINT_NAMES[j]}")
    axes2[0].set_title("Waist")
    axes2[0].legend(fontsize=7)
    axes2[0].grid(True, alpha=0.3)

    # Arm joints
    for j in range(15, 29):
        if j < len(G1_JOINT_NAMES):
            axes2[1].plot(time_axis, g1_targets[:, j], label=G1_JOINT_NAMES[j], alpha=0.7)
    axes2[1].set_title("Arms")
    axes2[1].legend(fontsize=6, ncol=3)
    axes2[1].grid(True, alpha=0.3)

    # Leg joints
    for j in range(0, 12):
        if j < len(G1_JOINT_NAMES):
            axes2[2].plot(time_axis, g1_targets[:, j], label=G1_JOINT_NAMES[j], alpha=0.7)
    axes2[2].set_title("Legs")
    axes2[2].set_xlabel("Time (s)")
    axes2[2].legend(fontsize=6, ncol=3)
    axes2[2].grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(str(traj_path), dpi=150)
    print(f"   ✅ Trajectory plot: {traj_path}")

    plt.close("all")
    print(f"\n{'=' * 60}")
    print(f"  Demo complete!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="SONIC × G1 × AMASS Demo")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--retarget", type=str,
                             help="Path to retargeted AMASS .npz (unimotion format)")
    input_group.add_argument("--synthetic", action="store_true", default=True,
                             help="Use synthetic motion (default)")

    parser.add_argument("--motion", default="walk", choices=["walk", "wave", "squat", "dance"])
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--model-dir", default="cross_embodiment_retarget_demo/checkpoints/sonic")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--vel-limit", type=float, default=10.0)
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--output-dir", default="cross_embodiment_retarget_demo/demos/output")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-anim-frames", type=int, default=300)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_demo(args)


if __name__ == "__main__":
    main()
