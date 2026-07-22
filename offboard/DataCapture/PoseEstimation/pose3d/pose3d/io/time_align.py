"""
pose3d.io.time_align — wraps Data_Preprocessing/align_audio.py (audio xcorr).

The upstream script's default video mapping is H.MOV/L.MP4/R.MP4; we write the
correct JSON for this data (H/L/R .MP4). Reference = H (delay 0). The script
produces <name>_aligned.MP4 for the non-reference views; the reference keeps its
original file. Reuses the script verbatim — no reimplementation.

Requires ffmpeg/ffprobe on PATH (align_audio.py shells out to them) and librosa
in this env (see environment.yml).
"""
from __future__ import annotations
import json
import subprocess
import os
import sys


def align_recording(recording_dir: str, views: list, reference: str, video_ext: str,
                    script_path: str, sr: int = 16000, run: bool = True) -> dict:
    """Run audio alignment for one recording.

    Returns {delays, aligned_paths, audio_alignment_path}.
    run=False only assembles paths (no execution). Uses sys.executable so the
    subprocess runs in THIS env (where librosa is installed), not a stray python.
    """
    os.makedirs(recording_dir, exist_ok=True)
    map_path = os.path.join(recording_dir, "_align_videos.json")
    with open(map_path, "w") as f:
        json.dump({v: f"{v}{video_ext}" for v in views}, f)

    out_json = os.path.join(recording_dir, "audio_alignment.json")

    if run:
        cmd = [sys.executable, os.path.abspath(script_path),
               "--videos", map_path,
               "--reference", reference,
               "--output", out_json,
               "--sr", str(sr),
               "--sync-video"]
        print("[time_align] running:", " ".join(cmd))
        # cwd = recording_dir so the script finds H/L/R.MP4 by relative name
        subprocess.run(cmd, check=True, cwd=os.path.abspath(recording_dir))

    delays = {reference: 0.0}
    if os.path.isfile(out_json):
        with open(out_json) as f:
            data = json.load(f)
        delays = {v: float(data.get(v, {}).get("delay_seconds", 0.0)) for v in views}

    aligned_paths = {}
    for v in views:
        if v == reference:
            aligned_paths[v] = os.path.join(recording_dir, f"{v}{video_ext}")
        else:
            aligned_paths[v] = os.path.join(recording_dir, f"{v}_aligned{video_ext}")
    return {"delays": delays, "aligned_paths": aligned_paths,
            "audio_alignment_path": out_json}
