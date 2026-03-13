"""
SONIC × G1 × AMASS Demo
========================
End-to-end demo: AMASS motion capture → SONIC ONNX retarget → G1 joint targets → 3D visualization.

Pipeline:
  1. Load AMASS data (real .npz or synthetic)
  2. Run SONIC encoder (smpl mode, 1762-dim → 64-dim tokens)
  3. Run SONIC decoder (994-dim → 29-dim G1 joint targets)
  4. Visualize side-by-side: SMPL input skeleton + G1 output skeleton

Usage:
    python -m cross_embodiment_retarget_demo.demos.sonic_g1_amass_demo --synthetic --duration 5
    python -m cross_embodiment_retarget_demo.demos.sonic_g1_amass_demo --amass path/to/motion.npz
    python -m cross_embodiment_retarget_demo.demos.sonic_g1_amass_demo --synthetic --motion dance --save-gif
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── G1 Skeleton Definition ────────────────────────────────────

# G1 joint names (29 DOF)
G1_JOINT_NAMES = [
    "torso_yaw", "torso_pitch", "torso_roll",                    # 0-2
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",     # 3-5
    "l_elbow", "l_wrist",                                        # 6-7
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",     # 8-10
    "r_elbow", "r_wrist",                                        # 11-12
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch",                   # 13-15
    "l_knee", "l_ankle_pitch", "l_ankle_roll",                   # 16-18
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch",                   # 19-21
    "r_knee", "r_ankle_pitch", "r_ankle_roll",                   # 22-24
    "l_hand_grip", "l_hand_wrist", "r_hand_grip", "r_hand_wrist",  # 25-28
]

# G1 simplified skeleton for visualization
# Each entry: (parent_keypoint_name, child_keypoint_name)
# We map 29 joint angles to approximate 3D keypoints
G1_KEYPOINTS = [
    "pelvis",
    "torso",
    "chest",
    "neck",
    "l_shoulder", "l_elbow", "l_wrist", "l_hand",
    "r_shoulder", "r_elbow", "r_wrist", "r_hand",
    "l_hip", "l_knee", "l_ankle", "l_foot",
    "r_hip", "r_knee", "r_ankle", "r_foot",
]

G1_BONES = [
    ("pelvis", "torso"),
    ("torso", "chest"),
    ("chest", "neck"),
    ("chest", "l_shoulder"), ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"), ("l_wrist", "l_hand"),
    ("chest", "r_shoulder"), ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"), ("r_wrist", "r_hand"),
    ("pelvis", "l_hip"), ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"), ("l_ankle", "l_foot"),
    ("pelvis", "r_hip"), ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"), ("r_ankle", "r_foot"),
]

# Default G1 keypoint positions (standing pose, Z-up, meters)
G1_DEFAULT_KEYPOINTS = {
    "pelvis":     np.array([0.0, 0.0, 0.80]),
    "torso":      np.array([0.0, 0.0, 0.92]),
    "chest":      np.array([0.0, 0.0, 1.05]),
    "neck":       np.array([0.0, 0.0, 1.15]),
    "l_shoulder": np.array([0.18, 0.0, 1.10]),
    "l_elbow":    np.array([0.18, 0.0, 0.84]),
    "l_wrist":    np.array([0.18, 0.0, 0.60]),
    "l_hand":     np.array([0.18, 0.0, 0.55]),
    "r_shoulder": np.array([-0.18, 0.0, 1.10]),
    "r_elbow":    np.array([-0.18, 0.0, 0.84]),
    "r_wrist":    np.array([-0.18, 0.0, 0.60]),
    "r_hand":     np.array([-0.18, 0.0, 0.55]),
    "l_hip":      np.array([0.09, 0.0, 0.80]),
    "l_knee":     np.array([0.09, 0.0, 0.45]),
    "l_ankle":    np.array([0.09, 0.0, 0.08]),
    "l_foot":     np.array([0.09, 0.08, 0.0]),
    "r_hip":      np.array([-0.09, 0.0, 0.80]),
    "r_knee":     np.array([-0.09, 0.0, 0.45]),
    "r_ankle":    np.array([-0.09, 0.0, 0.08]),
    "r_foot":     np.array([-0.09, 0.08, 0.0]),
}


def joint_angles_to_keypoints(
    joint_angles: np.ndarray,
    scale: float = 0.15,
) -> dict[str, np.ndarray]:
    """Convert 29 G1 joint angles to approximate 3D keypoint positions.

    This is a simplified kinematic mapping for visualization — not exact FK.
    Joint angles are mapped to displacements from the default standing pose.

    Args:
        joint_angles: (29,) joint targets in radians.
        scale: Sensitivity of angle → displacement mapping.

    Returns:
        Dict of keypoint_name → (3,) position.
    """
    q = joint_angles
    kp = {k: v.copy().astype(np.float64) for k, v in G1_DEFAULT_KEYPOINTS.items()}

    # Torso rotation
    torso_yaw, torso_pitch, torso_roll = q[0], q[1], q[2]
    kp["torso"][0] += torso_yaw * scale * 0.5
    kp["torso"][1] += torso_pitch * scale * 0.5
    kp["chest"][0] += torso_yaw * scale
    kp["chest"][1] += torso_pitch * scale
    kp["neck"][0] += torso_yaw * scale * 1.2
    kp["neck"][1] += torso_pitch * scale * 1.2

    # Left arm
    l_sp, l_sr, l_sy = q[3], q[4], q[5]
    l_elbow_angle, l_wrist_angle = q[6], q[7]

    kp["l_shoulder"][2] += -l_sp * scale * 0.3
    kp["l_shoulder"][0] += l_sr * scale * 0.3

    kp["l_elbow"][2] = kp["l_shoulder"][2] + (-0.26 + l_sp * scale * 0.5)
    kp["l_elbow"][0] = kp["l_shoulder"][0] + l_sr * scale * 0.5
    kp["l_elbow"][1] = kp["l_shoulder"][1] + l_sy * scale * 0.3

    elbow_bend = l_elbow_angle * scale
    kp["l_wrist"][2] = kp["l_elbow"][2] + (-0.24 + elbow_bend * 0.5)
    kp["l_wrist"][0] = kp["l_elbow"][0]
    kp["l_wrist"][1] = kp["l_elbow"][1] + elbow_bend * 0.3

    kp["l_hand"][2] = kp["l_wrist"][2] - 0.05
    kp["l_hand"][0] = kp["l_wrist"][0]
    kp["l_hand"][1] = kp["l_wrist"][1]

    # Right arm
    r_sp, r_sr, r_sy = q[8], q[9], q[10]
    r_elbow_angle, r_wrist_angle = q[11], q[12]

    kp["r_shoulder"][2] += -r_sp * scale * 0.3
    kp["r_shoulder"][0] += r_sr * scale * 0.3

    kp["r_elbow"][2] = kp["r_shoulder"][2] + (-0.26 + r_sp * scale * 0.5)
    kp["r_elbow"][0] = kp["r_shoulder"][0] + r_sr * scale * 0.5
    kp["r_elbow"][1] = kp["r_shoulder"][1] + r_sy * scale * 0.3

    elbow_bend_r = r_elbow_angle * scale
    kp["r_wrist"][2] = kp["r_elbow"][2] + (-0.24 + elbow_bend_r * 0.5)
    kp["r_wrist"][0] = kp["r_elbow"][0]
    kp["r_wrist"][1] = kp["r_elbow"][1] + elbow_bend_r * 0.3

    kp["r_hand"][2] = kp["r_wrist"][2] - 0.05
    kp["r_hand"][0] = kp["r_wrist"][0]
    kp["r_hand"][1] = kp["r_wrist"][1]

    # Left leg
    l_hy, l_hr, l_hp = q[13], q[14], q[15]
    l_knee_angle = q[16]
    l_ap, l_ar = q[17], q[18]

    kp["l_hip"][0] += l_hy * scale * 0.3
    kp["l_knee"][2] = kp["l_hip"][2] + (-0.35 + l_hp * scale * 0.2)
    kp["l_knee"][1] = kp["l_hip"][1] + l_hp * scale * 0.3
    kp["l_knee"][0] = kp["l_hip"][0] + l_hr * scale * 0.2

    kp["l_ankle"][2] = kp["l_knee"][2] + (-0.37 - l_knee_angle * scale * 0.3)
    kp["l_ankle"][1] = kp["l_knee"][1] + l_knee_angle * scale * 0.2
    kp["l_ankle"][0] = kp["l_knee"][0]

    kp["l_foot"][2] = max(0, kp["l_ankle"][2] - 0.08)
    kp["l_foot"][1] = kp["l_ankle"][1] + 0.08
    kp["l_foot"][0] = kp["l_ankle"][0]

    # Right leg
    r_hy, r_hr, r_hp = q[19], q[20], q[21]
    r_knee_angle = q[22]
    r_ap, r_ar = q[23], q[24]

    kp["r_hip"][0] += r_hy * scale * 0.3
    kp["r_knee"][2] = kp["r_hip"][2] + (-0.35 + r_hp * scale * 0.2)
    kp["r_knee"][1] = kp["r_hip"][1] + r_hp * scale * 0.3
    kp["r_knee"][0] = kp["r_hip"][0] + r_hr * scale * 0.2

    kp["r_ankle"][2] = kp["r_knee"][2] + (-0.37 - r_knee_angle * scale * 0.3)
    kp["r_ankle"][1] = kp["r_knee"][1] + r_knee_angle * scale * 0.2
    kp["r_ankle"][0] = kp["r_knee"][0]

    kp["r_foot"][2] = max(0, kp["r_ankle"][2] - 0.08)
    kp["r_foot"][1] = kp["r_ankle"][1] + 0.08
    kp["r_foot"][0] = kp["r_ankle"][0]

    return kp


# ── SMPL Skeleton Visualizer ──────────────────────────────────

SMPL_BONES = [
    (0, 1), (0, 2), (0, 3),     # pelvis → hips, spine
    (1, 4), (2, 5), (3, 6),     # hips → knees, spine → spine2
    (4, 7), (5, 8), (6, 9),     # knees → ankles, spine2 → spine3
    (7, 10), (8, 11),           # ankles → feet
    (9, 12), (9, 13), (9, 14),  # spine3 → neck, collars
    (12, 15),                    # neck → head
    (13, 16), (14, 17),          # collars → shoulders
    (16, 18), (17, 19),          # shoulders → elbows
    (18, 20), (19, 21),          # elbows → wrists
    (20, 22), (21, 23),          # wrists → hands
]


# ── Main Demo ─────────────────────────────────────────────────

def run_demo(args: argparse.Namespace):
    """Run the SONIC × G1 × AMASS end-to-end demo."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for GIF/video saving
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is required. Install: pip install onnxruntime-gpu")
        sys.exit(1)

    # Lazy imports to keep startup fast
    from ..src.amass_loader import SyntheticAMASS, load_amass
    from ..src.sonic_amass_bridge import SonicAMASSBridge

    # ── Load AMASS data ───────────────────────────────────────
    print("=" * 60)
    print("  SONIC × G1 × AMASS  End-to-End Demo")
    print("=" * 60)

    if args.amass:
        print(f"\n📂 Loading AMASS data: {args.amass}")
        motion = load_amass(args.amass, target_fps=args.fps)
    else:
        print(f"\n🎭 Generating synthetic motion: {args.motion}, {args.duration}s")
        synth = SyntheticAMASS(fps=args.fps)
        motion = synth.generate(motion_type=args.motion, duration=args.duration)

    print(f"   Frames: {motion.n_frames}, FPS: {motion.fps}, "
          f"Duration: {motion.duration:.1f}s")

    # ── Load SONIC ONNX models ────────────────────────────────
    model_dir = Path(args.model_dir).expanduser()
    encoder_path = model_dir / "model_encoder.onnx"
    decoder_path = model_dir / "model_decoder.onnx"

    if not encoder_path.exists() or not decoder_path.exists():
        print(f"\n❌ SONIC models not found in {model_dir}")
        print("   Expected: model_encoder.onnx, model_decoder.onnx")
        sys.exit(1)

    device = args.device
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    print(f"\n🧠 Loading SONIC ONNX models ({device})...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    encoder = ort.InferenceSession(
        str(encoder_path), sess_options=sess_opts, providers=providers,
    )
    decoder = ort.InferenceSession(
        str(decoder_path), sess_options=sess_opts, providers=providers,
    )

    # Verify providers
    enc_provider = encoder.get_providers()[0]
    dec_provider = decoder.get_providers()[0]
    print(f"   Encoder provider: {enc_provider}")
    print(f"   Decoder provider: {dec_provider}")

    # Verify shapes
    enc_input = encoder.get_inputs()[0]
    dec_input = decoder.get_inputs()[0]
    enc_output = encoder.get_outputs()[0]
    dec_output = decoder.get_outputs()[0]
    print(f"   Encoder: {enc_input.name} {enc_input.shape} → "
          f"{enc_output.name} {enc_output.shape}")
    print(f"   Decoder: {dec_input.name} {dec_input.shape} → "
          f"{dec_output.name} {dec_output.shape}")

    # ── Run inference loop ────────────────────────────────────
    bridge = SonicAMASSBridge(dt=1.0 / args.fps)

    n_frames = motion.n_frames
    if args.max_frames and args.max_frames < n_frames:
        n_frames = args.max_frames

    print(f"\n🚀 Running SONIC inference on {n_frames} frames...")

    # Storage for results
    smpl_positions_all = []     # (N, 24, 3)
    g1_joint_targets_all = []   # (N, 29)
    g1_keypoints_all = []       # (N, dict)
    encoder_tokens_all = []     # (N, 64)
    timing_stats = []

    for frame_idx in range(n_frames):
        t0 = time.perf_counter()

        # Pack encoder input
        enc_obs = bridge.pack_encoder_input(motion, frame_idx)
        t1 = time.perf_counter()

        # Run encoder
        enc_result = encoder.run(None, {enc_input.name: enc_obs})
        tokens = enc_result[0]  # (1, 64)
        t2 = time.perf_counter()

        # Pack decoder input
        dec_obs = bridge.pack_decoder_input(tokens)
        t3 = time.perf_counter()

        # Run decoder
        dec_result = decoder.run(None, {dec_input.name: dec_obs})
        action = dec_result[0].flatten()  # (29,)
        t4 = time.perf_counter()

        # Apply velocity limiting + joint clamping
        action = np.clip(action, -3.14, 3.14)

        # Step mock physics
        bridge.step_mock_physics(action, velocity_limit=args.vel_limit)
        t5 = time.perf_counter()

        # Store results
        smpl_positions_all.append(motion.joint_positions[frame_idx].copy())
        g1_joint_targets_all.append(bridge.joint_positions.copy())
        g1_keypoints_all.append(joint_angles_to_keypoints(bridge.joint_positions))
        encoder_tokens_all.append(tokens.flatten().copy())

        timing_stats.append({
            "pack_enc": (t1 - t0) * 1000,
            "run_enc": (t2 - t1) * 1000,
            "pack_dec": (t3 - t2) * 1000,
            "run_dec": (t4 - t3) * 1000,
            "physics": (t5 - t4) * 1000,
            "total": (t5 - t0) * 1000,
        })

        if frame_idx % 50 == 0 or frame_idx == n_frames - 1:
            stats = timing_stats[-1]
            print(f"   Frame {frame_idx:4d}/{n_frames}: "
                  f"enc={stats['run_enc']:.2f}ms  dec={stats['run_dec']:.2f}ms  "
                  f"total={stats['total']:.2f}ms  "
                  f"action_range=[{action.min():.3f}, {action.max():.3f}]")

    # ── Print summary statistics ──────────────────────────────
    print("\n" + "=" * 60)
    print("  Inference Statistics")
    print("=" * 60)

    # Skip first few frames (warmup)
    warmup = min(10, len(timing_stats) // 2)
    stats_arr = {
        k: np.array([s[k] for s in timing_stats[warmup:]])
        for k in timing_stats[0].keys()
    }

    for key, vals in stats_arr.items():
        print(f"   {key:10s}: mean={vals.mean():.3f}ms  "
              f"std={vals.std():.3f}ms  "
              f"p99={np.percentile(vals, 99):.3f}ms")

    fps_achieved = 1000.0 / stats_arr["total"].mean()
    print(f"\n   Effective FPS: {fps_achieved:.1f} (target: {args.fps})")

    # ── Analyze outputs ───────────────────────────────────────
    g1_targets = np.array(g1_joint_targets_all)  # (N, 29)
    tokens = np.array(encoder_tokens_all)        # (N, 64)

    print(f"\n   G1 joint targets range: [{g1_targets.min():.4f}, {g1_targets.max():.4f}]")
    print(f"   G1 joint targets std (per joint):")
    joint_stds = g1_targets.std(axis=0)
    active_joints = np.where(joint_stds > 0.01)[0]
    for j in active_joints:
        print(f"     Joint {j:2d} ({G1_JOINT_NAMES[j]:20s}): "
              f"std={joint_stds[j]:.4f}  "
              f"range=[{g1_targets[:, j].min():.3f}, {g1_targets[:, j].max():.3f}]")

    if len(active_joints) == 0:
        print("     ⚠ No active joints detected (all near-zero). "
              "The observation layout may need adjustment.")

    print(f"\n   Encoder token stats: mean={tokens.mean():.4f}, "
          f"std={tokens.std():.4f}, "
          f"range=[{tokens.min():.4f}, {tokens.max():.4f}]")

    # ── Generate visualization ────────────────────────────────
    print(f"\n🎬 Generating animation ({n_frames} frames)...")

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("SONIC × G1 × AMASS Demo", fontsize=14, fontweight="bold")

    # Left panel: SMPL input skeleton
    ax_smpl = fig.add_subplot(121, projection="3d")
    ax_smpl.set_title("SMPL Input (Motion Capture)")

    # Right panel: G1 output skeleton
    ax_g1 = fig.add_subplot(122, projection="3d")
    ax_g1.set_title("G1 Output (SONIC Retarget)")

    def set_axes_style(ax, x_range=(-0.6, 0.6), y_range=(-0.6, 0.6), z_range=(-0.1, 1.8)):
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_zlim(*z_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=15, azim=-60)

    # Compute SMPL centering offset (center pelvis at origin)
    smpl_arr = np.array(smpl_positions_all)  # (N, 24, 3)
    pelvis_mean = smpl_arr[:, 0, :].mean(axis=0)

    # Subsample frames for animation
    anim_step = max(1, n_frames // args.max_anim_frames)
    anim_indices = list(range(0, n_frames, anim_step))

    def update(anim_idx):
        frame = anim_indices[anim_idx]
        ax_smpl.cla()
        ax_g1.cla()

        # ── SMPL skeleton ──
        smpl_joints = smpl_arr[frame] - pelvis_mean + np.array([0, 0, 0.92])
        ax_smpl.set_title(f"SMPL Input (frame {frame})")
        set_axes_style(ax_smpl)

        # Draw bones
        for (j1, j2) in SMPL_BONES:
            p1, p2 = smpl_joints[j1], smpl_joints[j2]
            ax_smpl.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                "b-", linewidth=2, alpha=0.8,
            )
        # Draw joints
        ax_smpl.scatter(
            smpl_joints[:, 0], smpl_joints[:, 1], smpl_joints[:, 2],
            c="blue", s=20, alpha=0.9,
        )

        # ── G1 skeleton ──
        kp = g1_keypoints_all[frame]
        ax_g1.set_title(f"G1 Output (frame {frame})")
        set_axes_style(ax_g1)

        # Draw bones
        for (name1, name2) in G1_BONES:
            p1, p2 = kp[name1], kp[name2]
            ax_g1.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                "r-", linewidth=2.5, alpha=0.8,
            )
        # Draw joints
        pts = np.array([kp[name] for name in G1_KEYPOINTS])
        ax_g1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="red", s=25, alpha=0.9)

        return []

    n_anim = len(anim_indices)
    anim = FuncAnimation(
        fig, update, frames=n_anim,
        interval=1000.0 / min(args.fps, 30),  # Cap display FPS at 30
        blit=False,
    )

    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_gif:
        gif_path = output_dir / f"sonic_g1_amass_{motion.source}.gif"
        print(f"   Saving GIF: {gif_path}")
        writer = PillowWriter(fps=min(args.fps, 25))
        anim.save(str(gif_path), writer=writer, dpi=80)
        print(f"   ✅ Saved: {gif_path} ({gif_path.stat().st_size / 1024:.0f} KB)")

    if args.save_mp4:
        mp4_path = output_dir / f"sonic_g1_amass_{motion.source}.mp4"
        print(f"   Saving MP4: {mp4_path}")
        try:
            anim.save(str(mp4_path), writer="ffmpeg", fps=min(args.fps, 30), dpi=100)
            print(f"   ✅ Saved: {mp4_path}")
        except Exception as e:
            print(f"   ⚠ MP4 save failed (ffmpeg required): {e}")

    # Also save a static plot of joint trajectories
    traj_path = output_dir / f"sonic_g1_joint_trajectories_{motion.source}.png"
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig2.suptitle("G1 Joint Trajectories from SONIC Retarget", fontsize=13)

    time_axis = np.arange(g1_targets.shape[0]) / args.fps

    # Plot torso joints
    for j in range(3):
        axes2[0].plot(time_axis, g1_targets[:, j], label=G1_JOINT_NAMES[j])
    axes2[0].set_ylabel("Angle (rad)")
    axes2[0].set_title("Torso Joints")
    axes2[0].legend(fontsize=8)
    axes2[0].grid(True, alpha=0.3)

    # Plot arm joints
    for j in range(3, 13):
        axes2[1].plot(time_axis, g1_targets[:, j], label=G1_JOINT_NAMES[j], alpha=0.7)
    axes2[1].set_ylabel("Angle (rad)")
    axes2[1].set_title("Arm Joints")
    axes2[1].legend(fontsize=7, ncol=2)
    axes2[1].grid(True, alpha=0.3)

    # Plot leg joints
    for j in range(13, 25):
        axes2[2].plot(time_axis, g1_targets[:, j], label=G1_JOINT_NAMES[j], alpha=0.7)
    axes2[2].set_ylabel("Angle (rad)")
    axes2[2].set_xlabel("Time (s)")
    axes2[2].set_title("Leg Joints")
    axes2[2].legend(fontsize=7, ncol=2)
    axes2[2].grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(str(traj_path), dpi=150)
    print(f"   ✅ Joint trajectory plot: {traj_path}")

    plt.close("all")

    print(f"\n{'=' * 60}")
    print(f"  Demo complete!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="SONIC × G1 × AMASS End-to-End Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic walking motion, 5 seconds, save GIF
  python -m cross_embodiment_retarget_demo.demos.sonic_g1_amass_demo --synthetic --duration 5 --save-gif

  # Synthetic dance, 10 seconds
  python -m cross_embodiment_retarget_demo.demos.sonic_g1_amass_demo --synthetic --motion dance --duration 10 --save-gif

  # Real AMASS data
  python -m cross_embodiment_retarget_demo.demos.sonic_g1_amass_demo --amass path/to/CMU/01/01_01_poses.npz --save-gif

  # CPU-only mode
  python -m cross_embodiment_retarget_demo.demos.sonic_g1_amass_demo --synthetic --device cpu --save-gif
""",
    )

    # Input source
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--amass", type=str, help="Path to AMASS .npz file")
    input_group.add_argument(
        "--synthetic", action="store_true", default=True,
        help="Use synthetic motion data (default)",
    )

    # Motion parameters
    parser.add_argument(
        "--motion", type=str, default="walk",
        choices=["walk", "wave", "squat", "dance"],
        help="Synthetic motion type (default: walk)",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Duration in seconds (default: 5.0)",
    )
    parser.add_argument("--fps", type=float, default=50.0, help="Target FPS (default: 50)")

    # Model
    parser.add_argument(
        "--model-dir", type=str,
        default="cross_embodiment_retarget_demo/checkpoints/sonic",
        help="Path to SONIC ONNX models directory",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Inference device (default: cuda)",
    )

    # Physics
    parser.add_argument(
        "--vel-limit", type=float, default=10.0,
        help="Maximum joint velocity in rad/s (default: 10.0)",
    )

    # Output
    parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF")
    parser.add_argument("--save-mp4", action="store_true", help="Save animation as MP4")
    parser.add_argument(
        "--output-dir", type=str,
        default="cross_embodiment_retarget_demo/demos/output",
        help="Output directory for saved files",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--max-anim-frames", type=int, default=300,
        help="Maximum frames in animation (subsamples if needed)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_demo(args)


if __name__ == "__main__":
    main()
