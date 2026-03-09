"""
GenRobot Dataset Downloader & Synthetic Data Generator
=======================================================
Dataset: genrobot2025/10Kh-RealOmin-OpenData
Format: MCAP (multi-sensor) -> H5 (training-ready)

Usage:
  python download_data.py --mode synthetic          # no token needed
  python download_data.py --mode real --hf-token YOUR_TOKEN
  python download_data.py --mode verify
  python download_data.py --mode sync
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from huggingface_hub import HfApi, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
GENROBOT_DIR  = DATA_DIR / "genrobot_open_dataset"
HF_DATASET_ID = "genrobot2025/10Kh-RealOmin-OpenData"

SKILLS = [
    ("Folding_Clothes_and_Zipper_Operations", ["fold_and_store_clothes", "zip_clothes"]),
    ("Cooking_and_Kitchen_Clean",             ["clean_container", "unscrew_bottle_cap_and_pour", "clean_bowl"]),
    ("Organize_Clutter",                      ["fold_and_store_shopping_bag", "fold_towel",
                                               "desktop_object_sorting", "drawer_to_take_items"]),
    ("Shoes_Handling",                        ["lace_up_shoes_with_both_hands", "organize_scattered_shoes"]),
    ("Clutter_Tidy-Up",                       ["irregular_object_clutter", "flexible_grasping_and_sorting",
                                               "carton_sorting_clutter", "small_object_storage"]),
]

# ── Synthetic frame / trajectory generators ───────────────────────────────────

def _make_frame(idx, h=480, w=640):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        r = int(180 * (1 - y / h) + 60)
        g = int(160 * (1 - y / h) + 50)
        b = int(140 * (1 - y / h) + 40)
        img[y, :] = [b, g, r]
    cv2.rectangle(img, (0, h // 2), (w, h), (100, 120, 140), -1)
    ox = int(w * 0.3 + 50 * np.sin(idx * 0.1))
    oy = int(h * 0.65)
    cv2.rectangle(img, (ox - 20, oy - 20), (ox + 20, oy + 20), (60, 120, 200), -1)
    gx = int(w * 0.3 + 50 * np.sin(idx * 0.1))
    gy = int(h * 0.5 - 20 * np.cos(idx * 0.05))
    cv2.circle(img, (gx, gy), 12, (200, 50, 50), -1)
    cv2.line(img, (gx, gy), (ox, oy), (200, 200, 200), 2)
    cv2.putText(img, f"Frame {idx:04d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def _eef_traj(n):
    t = np.linspace(0, 2 * np.pi, n)
    px = 0.3 + 0.1 * np.sin(t)
    py = 0.05 * np.cos(t * 2)
    pz = 0.5 + 0.15 * np.abs(np.sin(t))
    qx = np.zeros(n)
    qy = np.sin(t * 0.5) * 0.1
    qz = np.zeros(n)
    qw = np.sqrt(1 - qy ** 2)
    gr = 0.08 * np.abs(np.sin(t * 0.5))
    return np.column_stack([px, py, pz, qx, qy, qz, qw, gr]).astype(np.float32)


def _imu(n):
    t = np.linspace(0, 4 * np.pi, n)
    av = np.column_stack([0.01 * np.sin(t), 0.01 * np.cos(t), 0.005 * np.sin(t * 2)])
    la = np.column_stack([0.1 * np.sin(t * 3), 0.1 * np.cos(t * 3), 9.81 + 0.05 * np.sin(t)])
    return np.hstack([av, la]).astype(np.float32)

# ── H5 file generator ─────────────────────────────────────────────────────────

def generate_h5(path, n=150, skill="pick_and_place"):
    if not H5PY_AVAILABLE:
        print("h5py not installed: pip install h5py")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Generating {n}-frame H5: {path.name}")
    with h5py.File(str(path), "w") as f:
        f.attrs.update({
            "skill": skill, "num_frames": n, "fps": 30,
            "source": "synthetic", "dataset": HF_DATASET_ID,
        })
        obs = f.create_group("observations")
        cam = obs.create_group("cameras")
        dt = h5py.special_dtype(vlen=np.dtype("uint8"))
        ds = cam.create_dataset("mid_fisheye_color", shape=(n,), dtype=dt)
        for i in range(n):
            _, enc = cv2.imencode(".jpg", _make_frame(i), [cv2.IMWRITE_JPEG_QUALITY, 80])
            ds[i] = np.frombuffer(enc.tobytes(), dtype=np.uint8)
        traj = _eef_traj(n)
        obs.create_dataset("eef_pos", data=traj, compression="gzip", compression_opts=4)
        obs.create_dataset("imu",     data=_imu(n), compression="gzip", compression_opts=4)
        tac = obs.create_group("tactile")
        tac.create_dataset("left",  data=(np.random.rand(n, 12, 8) * 0.5).astype(np.float32),
                           compression="gzip", compression_opts=4)
        tac.create_dataset("right", data=(np.random.rand(n, 12, 8) * 0.5).astype(np.float32),
                           compression="gzip", compression_opts=4)
        f.create_dataset("action", data=traj, compression="gzip", compression_opts=4)
    sz = path.stat().st_size / 1024
    print(f"  done  {path.name} ({sz:.1f} KB)")
    return True

# ── Synthetic dataset builder ─────────────────────────────────────────────────

def generate_synthetic_dataset(eps_per_skill=3):
    print("\n[SYNTHETIC] Generating GenRobot-format synthetic dataset...")
    GENROBOT_DIR.mkdir(parents=True, exist_ok=True)
    all_eps = []
    skill_stats = {}
    for cat, skills in SKILLS:
        cat_dir = GENROBOT_DIR / cat
        cat_dir.mkdir(exist_ok=True)
        for skill in skills:
            sk_dir = cat_dir / skill
            sk_dir.mkdir(exist_ok=True)
            sk_eps = []
            for i in range(eps_per_skill):
                n = int(np.random.randint(100, 250))
                h5_path = sk_dir / f"ep_{i:04d}.h5"
                if generate_h5(h5_path, n, skill):
                    ep = {
                        "id": f"ep_{cat}_{skill}_{i:04d}",
                        "datasetId": "genrobot_10kh",
                        "skill": skill,
                        "category": cat,
                        "name": f"{skill} #{i}",
                        "description": f"Synthetic episode for {skill}",
                        "h5_path": str(h5_path),
                        "frameCount": n,
                        "duration": round(n / 30, 2),
                        "fps": 30,
                        "sensors": ["mid_fisheye_color", "imu", "tactile_left", "tactile_right", "eef_pose"],
                        "robot": "GenDAS Gripper v3",
                        "bimanual": False,
                        "createdAt": datetime.utcnow().isoformat(),
                        "metadata": {
                            "source": "synthetic",
                            "stage": "2" if cat == "Clutter_Tidy-Up" else "1",
                            "format": "h5",
                            "original_format": "mcap",
                            "resolution": "640x480",
                            "camera_type": "Large FOV Fisheye",
                            "has_imu": True,
                            "has_tactile": True,
                        },
                    }
                    sk_eps.append(ep)
                    all_eps.append(ep)
            skill_stats[skill] = {"episodes": len(sk_eps), "category": cat}
            print(f"  [{cat}] {skill}: {len(sk_eps)} episodes")

    meta = {
        "id": "genrobot_10kh",
        "name": "10Kh-RealOmin-OpenData (Synthetic Preview)",
        "source": HF_DATASET_ID,
        "description": ("Largest open embodied intelligence dataset. "
                        "10,000+ hours of real household robot manipulation. "
                        "Synthetic preview — real data requires a HuggingFace token."),
        "format": "h5",
        "original_format": "mcap",
        "resolution": "1600x1296",
        "real_fps": 30,
        "robot_type": "GenDAS Gripper",
        "bimanual": True,
        "sensors": {
            "camera":  "Large FOV Fisheye (H.264)",
            "imu":     "6-axis IMU",
            "tactile": "1mm tactile array",
            "pose":    "VIO EEF 7-DOF",
            "gripper": "Magnetic encoder",
        },
        "stage1": {"skills": 12, "hours": 950,  "clips": 39761, "size_tb": 3.45},
        "stage2": {"skills":  4, "hours": 653,  "clips": 36267, "size_tb": 1.92},
        "total_real_hours": 10000,
        "households": 3000,
        "unique_targets": 10000,
        "skill_stats": skill_stats,
        "episodes": all_eps,
        "episodeCount": len(all_eps),
        "syntheticEpisodes": len(all_eps),
        "realEpisodes": 0,
        "datakit_repo": "https://github.com/genrobot-ai/das-datakit",
        "generated_at": datetime.utcnow().isoformat(),
    }
    (GENROBOT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nGenerated {len(all_eps)} synthetic episodes -> {GENROBOT_DIR}/metadata.json")
    return meta

# ── Real HuggingFace download ─────────────────────────────────────────────────

def download_real_dataset(hf_token, max_files=5, skill_filter=None):
    if not HF_AVAILABLE:
        print("huggingface_hub not installed: pip install huggingface_hub")
        return False
    print(f"\n[REAL] Connecting to {HF_DATASET_ID} ...")
    print("WARNING: Dataset is GATED. You must accept terms on the HuggingFace page first.\n")
    api = HfApi(token=hf_token)
    try:
        files = list(api.list_repo_files(
            repo_id=HF_DATASET_ID, repo_type="dataset", token=hf_token))
        mcap_files = [f for f in files if f.endswith(".mcap")]
        if skill_filter:
            mcap_files = [f for f in mcap_files if skill_filter in f]
        print(f"Found {len(mcap_files)} MCAP files")
        if not mcap_files:
            print("No MCAP files found. Check token and accepted terms.")
            return False
        real_dir = GENROBOT_DIR / "real_mcap"
        real_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []
        for fp in mcap_files[:max_files]:
            try:
                print(f"  Downloading: {fp}")
                local = hf_hub_download(
                    repo_id=HF_DATASET_ID, filename=fp,
                    repo_type="dataset", token=hf_token,
                    local_dir=str(real_dir),
                )
                downloaded.append(local)
                print(f"  OK  {Path(local).name}")
                _mcap_to_h5(Path(local))
            except Exception as e:
                print(f"  FAIL  {fp}: {e}")
        return len(downloaded) > 0
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nFix:")
        print("  1) Accept terms at https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData")
        print("  2) Get token at   https://huggingface.co/settings/tokens")
        print("  3) Run: python download_data.py --mode real --hf-token YOUR_TOKEN")
        return False


def _mcap_to_h5(mcap_path):
    dk = BASE_DIR / "das-datakit"
    if not dk.exists():
        return
    h5 = mcap_path.with_suffix(".h5")
    result = subprocess.run(
        [sys.executable, str(dk / "mcap_to_h5.py"),
         "--mcap-file", str(mcap_path), "--out_path", str(h5)],
        capture_output=True, text=True,
    )
    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"  [{status}] H5: {h5.name}")
    if result.returncode != 0:
        print(result.stderr[:400])

# ── Platform DB sync ──────────────────────────────────────────────────────────

def sync_to_platform_db():
    mp = GENROBOT_DIR / "metadata.json"
    if not mp.exists():
        print("ERROR: Run --mode synthetic first to generate metadata.json")
        return

    meta = json.loads(mp.read_text())

    # --- datasets.json ---
    dp = DATA_DIR / "datasets.json"
    ds_list = json.loads(dp.read_text()) if dp.exists() else []
    ds_list = [d for d in ds_list if d.get("id") != "genrobot_10kh"]
    total_frames = sum(e["frameCount"] for e in meta["episodes"])
    total_sz_mb = sum(
        Path(e["h5_path"]).stat().st_size if Path(e["h5_path"]).exists() else 0
        for e in meta["episodes"]
    ) // (1024 * 1024)
    ds_list.append({
        "id": "genrobot_10kh",
        "name": "10Kh-RealOmin-OpenData",
        "description": meta["description"],
        "format": "lerobot",
        "robotType": meta["robot_type"],
        "source": meta["source"],
        "taskDescription": "Household robot manipulation: folding, cooking, cleaning, organizing",
        "environment": {"type": "household", "stages": ["Stage1", "Stage2"]},
        "createdAt":   meta["generated_at"],
        "updatedAt":   meta["generated_at"],
        "episodeCount": meta["episodeCount"],
        "frameCount":   total_frames,
        "size": total_sz_mb,
        "version": "1.0.0",
        "license": "CC-BY-SA-4.0",
        "sensorConfig": {
            "name": "DAS Gripper Sensor Suite",
            "sensors": [
                {"type": "camera",  "name": "mid_fisheye",  "location": "gripper_center"},
                {"type": "imu",     "name": "imu_6axis",    "location": "gripper_body"},
                {"type": "tactile", "name": "tactile_left", "location": "finger_left"},
                {"type": "tactile", "name": "tactile_right","location": "finger_right"},
                {"type": "encoder", "name": "magnetic_enc", "location": "gripper_joint"},
            ],
        },
        "skills": [s for _, skills in SKILLS for s in skills],
        "syntheticPreview": True,
        "realDataStats": meta.get("stage1", {}),
    })
    dp.write_text(json.dumps(ds_list, indent=2))
    print(f"OK  datasets.json: {len(ds_list)} datasets")

    # --- episodes.json ---
    ep_path = DATA_DIR / "episodes.json"
    eps = json.loads(ep_path.read_text()) if ep_path.exists() else []
    eps = [e for e in eps if e.get("datasetId") != "genrobot_10kh"]
    eps.extend(meta["episodes"])
    ep_path.write_text(json.dumps(eps, indent=2))
    print(f"OK  episodes.json: {len(eps)} total ({len(meta['episodes'])} genrobot)")

    # --- annotations.json seed ---
    ann_p = DATA_DIR / "annotations.json"
    anns = []
    if ann_p.exists():
        try:
            anns = json.loads(ann_p.read_text())
        except Exception:
            anns = []
    anns = [a for a in anns if not a.get("id", "").startswith("ann_gr_seed_")]
    for ep in meta["episodes"][:5]:
        anns.append({
            "id":        f"ann_gr_seed_{ep['id']}",
            "episodeId": ep["id"],
            "datasetId": "genrobot_10kh",
            "type":      "label",
            "label":     ep["skill"],
            "confidence": 1.0,
            "annotator": "metadata",
            "verified":  True,
            "createdAt": meta["generated_at"],
            "updatedAt": meta["generated_at"],
        })
    ann_p.write_text(json.dumps(anns, indent=2))
    print(f"OK  annotations.json: {len(anns)} annotations")
    print("DB synced — refresh UI to see the dataset.")

# ── Verification ──────────────────────────────────────────────────────────────

def verify_dataset():
    mp = GENROBOT_DIR / "metadata.json"
    if not mp.exists():
        print("ERROR: No dataset. Run --mode synthetic first.")
        return False
    if not H5PY_AVAILABLE:
        print("h5py not installed: pip install h5py")
        return False
    meta = json.loads(mp.read_text())
    ok = 0
    fail = 0
    print(f"\n[VERIFY] Checking {meta['episodeCount']} episodes ...")
    for ep in meta["episodes"]:
        p = Path(ep["h5_path"])
        if not p.exists():
            print(f"  MISSING  {p}")
            fail += 1
            continue
        try:
            with h5py.File(str(p), "r") as f:
                assert "observations/cameras/mid_fisheye_color" in f, "no camera"
                assert "observations/eef_pos" in f, "no eef_pos"
                assert f["observations/eef_pos"].shape[1] == 8, "eef_pos shape mismatch"
            ok += 1
        except Exception as e:
            print(f"  FAIL  {p.name}: {e}")
            fail += 1
    print(f"\n  OK={ok}  FAIL={fail}")
    return fail == 0

# ── Extract frames from video ─────────────────────────────────────────────────

def extract_frames(video_path, output_dir, frame_rate=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return 0
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    interval = max(1, fps // frame_rate)
    fc = sc = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fc % interval == 0:
            cv2.imwrite(str(output_dir / f"frame_{sc:04d}.jpg"), frame)
            sc += 1
        fc += 1
    cap.release()
    print(f"Extracted {sc} frames -> {output_dir}")
    return sc

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GenRobot Dataset Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py --mode synthetic --episodes 3
  python download_data.py --mode real --hf-token hf_xxx --max-files 5
  python download_data.py --mode verify
  python download_data.py --mode sync
""",
    )
    p.add_argument("--mode", choices=["synthetic", "real", "both", "verify", "sync"],
                   default="synthetic")
    p.add_argument("--hf-token",  type=str, default=None,
                   help="HuggingFace read token (required for --mode real)")
    p.add_argument("--max-files", type=int, default=5,
                   help="Max MCAP files to download (default 5)")
    p.add_argument("--skill",     type=str, default=None,
                   help="Filter files by skill name substring")
    p.add_argument("--episodes",  type=int, default=3,
                   help="Episodes per skill for synthetic mode (default 3)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode in ("synthetic", "both"):
        generate_synthetic_dataset(args.episodes)
        sync_to_platform_db()

    if args.mode in ("real", "both"):
        if not args.hf_token:
            print("ERROR: --hf-token is required for real download")
            print("  Get a token at https://huggingface.co/settings/tokens")
            sys.exit(1)
        if download_real_dataset(args.hf_token, args.max_files, args.skill):
            sync_to_platform_db()

    if args.mode == "verify":
        verify_dataset()

    if args.mode == "sync":
        sync_to_platform_db()

    print("\nDone.")

