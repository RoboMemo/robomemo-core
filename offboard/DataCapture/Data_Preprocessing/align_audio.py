#!/usr/bin/env python3
"""
Audio alignment script for multi-camera synchronization.

Aligns audio tracks from multiple video files to a reference track
using energy envelope and normalized cross-correlation.

Usage:
    python3 align_audio.py [--output OUTPUT_JSON] [--sr SAMPLE_RATE] [--reference REF_NAME]

Example:
    python3 align_audio.py --reference H --output audio_alignment.json
"""

import subprocess
import numpy as np
import librosa
from scipy import signal
import json
from pathlib import Path
import argparse


def extract_audio(video_file: str, audio_output: str) -> None:
    """Extract audio from video file."""
    cmd = f'ffmpeg -i "{video_file}" -q:a 9 -n "{audio_output}" -loglevel error'
    subprocess.run(cmd, shell=True, check=True)


def load_audio(audio_file: str, sr: int = 16000) -> tuple:
    """Load audio file and return (audio_data, sample_rate)."""
    y, sr = librosa.load(audio_file, sr=sr)
    return y, sr


def compute_energy_envelope(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute energy envelope using mel-spectrogram.

    Args:
        audio: Audio time series
        sr: Sample rate

    Returns:
        Energy envelope (1D array)
    """
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    env = np.mean(S, axis=0)
    return env


def find_alignment(ref_envelope: np.ndarray, test_envelope: np.ndarray,
                   hop_length: int = 512, sr: int = 16000, max_delay_seconds: float = 10.0) -> dict:
    """
    Find time delay between two signals using normalized cross-correlation.

    Args:
        ref_envelope: Reference signal envelope
        test_envelope: Test signal envelope
        hop_length: Hop length in samples for mel-spectrogram frame
        sr: Sample rate
        max_delay_seconds: Maximum allowed delay in seconds (default: 10s). Limits search window
                          to avoid false peaks from videos of different lengths.

    Returns:
        Dictionary with delay information:
        - delay_frames: Delay in frame count
        - delay_samples: Delay in audio samples
        - delay_seconds: Delay in seconds
        - correlation: Cross-correlation value
    """
    # Normalize both signals
    ref_norm = (ref_envelope - np.mean(ref_envelope)) / (np.std(ref_envelope) + 1e-8)
    test_norm = (test_envelope - np.mean(test_envelope)) / (np.std(test_envelope) + 1e-8)

    # Compute normalized cross-correlation
    correlation = signal.correlate(ref_norm, test_norm, mode='full')
    lags = signal.correlation_lags(len(ref_norm), len(test_norm), mode='full')

    # Restrict search to reasonable delay window to avoid false peaks from videos of different lengths
    max_lag_frames = int(max_delay_seconds * sr / hop_length)
    mask = (lags >= -max_lag_frames) & (lags <= max_lag_frames)

    # Find peak within restricted window
    abs_correlation = np.abs(correlation[mask])
    if len(abs_correlation) == 0:
        # Fallback if mask is empty
        max_corr_idx = np.argmax(np.abs(correlation))
    else:
        max_corr_idx_in_mask = np.argmax(abs_correlation)
        max_corr_idx = np.where(mask)[0][max_corr_idx_in_mask]

    delay_frames = lags[max_corr_idx]
    delay_samples = delay_frames * hop_length
    delay_seconds = delay_samples / sr

    return {
        'delay_frames': int(delay_frames),
        'delay_samples': int(delay_samples),
        'delay_seconds': float(delay_seconds),
        'correlation': float(correlation[max_corr_idx])
    }


def align_videos(video_dict: dict, reference: str = 'H', output_json: str = 'audio_alignment.json',
                 sr: int = 16000) -> dict:
    """
    Align multiple video files by their audio tracks.

    Args:
        video_dict: Dictionary mapping camera names to video file paths
        reference: Name of reference camera (default: 'H')
        output_json: Output JSON file path
        sr: Sample rate for audio processing

    Returns:
        Alignment results dictionary
    """
    print(f"Step 1: Extracting audio from {len(video_dict)} videos...")
    audio_files = {}
    for name, video in video_dict.items():
        audio_file = f'audio_{name}.wav'
        extract_audio(video, audio_file)
        audio_files[name] = audio_file
        print(f"  ✓ {name}: {audio_file}")

    # Load audio
    print(f"\nStep 2: Loading audio (downsampling to {sr}Hz)...")
    audios = {}
    for name, audio_file in audio_files.items():
        y, sr = load_audio(audio_file, sr=sr)
        audios[name] = y
        print(f"  ✓ {name}: {len(y)} samples ({len(y)/sr:.2f}s)")

    # Compute energy envelopes
    print("\nStep 3: Computing energy envelopes...")
    envelopes = {}
    hop_length = 512
    for name, y in audios.items():
        env = compute_energy_envelope(y, sr)
        envelopes[name] = env
        print(f"  ✓ {name}: {len(env)} frames")

    # Align to reference
    print(f"\nStep 4: Aligning to reference '{reference}'...")
    ref_envelope = envelopes[reference]

    alignment_results = {
        reference: {
            'delay_samples': 0,
            'delay_seconds': 0.0,
            'method': 'normalized_xcorr',
            'hop_length': hop_length
        }
    }

    for name in video_dict.keys():
        if name != reference:
            test_envelope = envelopes[name]
            result = find_alignment(ref_envelope, test_envelope, hop_length, sr)
            alignment_results[name] = result

            direction = "behind" if result['delay_frames'] > 0 else "ahead of"
            print(f"  ✓ {name}: {result['delay_seconds']:+.4f}s ({result['delay_frames']:+d} frames, {direction} {reference})")

    # Save results
    print(f"\nStep 5: Saving results to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(alignment_results, f, indent=2)

    # Cleanup
    print("\nStep 6: Cleaning up temporary files...")
    for audio_file in audio_files.values():
        Path(audio_file).unlink()
        print(f"  ✓ Removed {audio_file}")

    print("\n✓ Audio alignment complete!")
    print("\nAlignment Offsets (relative to reference):")
    for name, result in alignment_results.items():
        if 'delay_seconds' in result:
            print(f"  {name}: {result['delay_seconds']:+.4f}s")

    return alignment_results


def get_video_duration(video_file: str) -> float:
    """
    Get video duration in seconds using ffprobe.

    Args:
        video_file: Path to video file

    Returns:
        Duration in seconds
    """
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1:nokey=1 "{video_file}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def get_video_fps(video_file: str) -> float:
    """
    Get video frame rate using ffprobe.

    Args:
        video_file: Path to video file

    Returns:
        Frame rate (fps)
    """
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "{video_file}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    fps_str = result.stdout.strip()
    # Handle frame rate as fraction (e.g., "30000/1001")
    if '/' in fps_str:
        num, den = fps_str.split('/')
        return float(num) / float(den)
    return float(fps_str)


def get_video_resolution(video_file: str) -> tuple:
    """
    Get video resolution (width, height) using ffprobe.

    Args:
        video_file: Path to video file

    Returns:
        Tuple of (width, height)
    """
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "{video_file}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    parts = result.stdout.strip().split(',')
    width, height = int(parts[0]), int(parts[1])
    return width, height


def ensure_camera_metadata(video: str):
    """
    Ensure a `<video>_metadata.json` exists for a DJI camera by decoding its
    embedded djmd stream with telemetry-parser (the engine Gyroflow uses). This
    is the authoritative way to recover Default (camera params) + Lens
    (calibration) + per-frame Quaternion (IMU attitude, already in the correct
    coordinate frame) as a per-frame JSON array.

    ffmpeg's mp4 muxer cannot re-write DJI djmd/dbgi streams into a new container,
    so the JSON is the metadata carrier. If the JSON already exists it is reused
    (so user-provided / previously generated files are not overwritten).

    Returns the Path to the metadata JSON, or None if telemetry-parser is not
    installed or the file isn't a supported DJI clip.
    """
    meta_path = Path(video).with_name(f"{Path(video).stem}_metadata.json")
    if meta_path.exists():
        return meta_path
    try:
        import telemetry_parser
    except ImportError:
        print(f"  ℹ telemetry-parser not installed; cannot generate {meta_path.name} "
              f"(pip install telemetry-parser, or build from github master for DJI Osmo)")
        return None
    try:
        data = telemetry_parser.Parser(video).telemetry()
        meta_path.write_text(json.dumps(data))
        print(f"  ✓ Generated {meta_path.name} ({len(data)} frames)")
        return meta_path
    except Exception as e:
        print(f"  ℹ Could not generate metadata for {video}: {e}")
        return None


def copy_metadata_with_trim(src_video: str, aligned_video: str,
                            trim_start_sec: float, trim_duration_sec: float) -> None:
    """
    Copy the source *_metadata.json to the aligned video's name, preserving the
    DJI static calibration (Default/Lens) and time-syncing the per-frame array
    and Quaternion samples to the trimmed timeline.

    The DJI metadata JSON is a per-frame array (len == #frames == djmd packets).
    Each element carries a Quaternion.Data window of ~1000 Hz IMU samples with
    absolute 't' timestamps in milliseconds; element[0] additionally holds the
    static Default/Lens calibration. After trimming the video, the array is
    sliced to match and the 't' values are re-based to zero so they stay in sync
    with both the trimmed video frames and the in-container djmd stream.
    """
    stem = Path(src_video).stem
    src_meta = Path(src_video).with_name(f"{stem}_metadata.json")
    if not src_meta.exists():
        print(f"  ℹ No {src_meta.name} found, skipping metadata copy")
        return

    meta = json.loads(src_meta.read_text())

    # Unknown / non-array structure: copy verbatim without time-sync.
    if not isinstance(meta, list) or not meta:
        out_meta = Path(aligned_video).with_name(f"{Path(aligned_video).stem}_metadata.json")
        out_meta.write_text(json.dumps(meta))
        print(f"  ✓ Copied metadata (passthrough) → {out_meta.name}")
        return

    fps = get_video_fps(src_video)
    drop_frames = int(round(trim_start_sec * fps))
    keep_frames = int(round(trim_duration_sec * fps))
    trim_offset_ms = trim_start_sec * 1000.0  # 't' unit is milliseconds (IMU @ 1000 Hz)

    # Static calibration lives in element[0]; preserve it across the trim.
    static = {}
    if isinstance(meta[0], dict):
        for k in ("Default", "Lens"):
            if k in meta[0]:
                static[k] = meta[0][k]

    # Slice the per-frame array to match the trimmed video, then zero-base t.
    trimmed = meta[drop_frames: drop_frames + keep_frames]
    for elem in trimmed:
        if not isinstance(elem, dict):
            continue
        data = (elem.get("Quaternion") or {}).get("Data")
        if isinstance(data, list):
            for s in data:
                if isinstance(s, dict) and "t" in s:
                    s["t"] = round(float(s["t"]) - trim_offset_ms, 6)

    # Re-attach static calibration to the new first element so it isn't lost.
    if trimmed and isinstance(trimmed[0], dict):
        merged = dict(trimmed[0])
        for k, v in static.items():
            merged.setdefault(k, v)
        trimmed[0] = merged

    out_meta = Path(aligned_video).with_name(f"{Path(aligned_video).stem}_metadata.json")
    out_meta.write_text(json.dumps(trimmed))
    print(f"  ✓ Copied+synced metadata ({len(trimmed)} frames) → {out_meta.name}")


def sync_video_to_reference(alignment_json: str, video_dict: dict,
                            reference: str = 'H',
                            output_suffix: str = '_aligned') -> dict:
    """
    Synchronize video files based on audio alignment results.

    This function reads the alignment JSON and generates new video files
    where L and R cameras are temporally aligned to the reference camera (H).

    Args:
        alignment_json: Path to alignment JSON file
        video_dict: Dictionary mapping camera names to video file paths
        reference: Name of reference camera
        output_suffix: Suffix for output aligned video files

    Returns:
        Dictionary with paths to aligned video files
    """
    print(f"Step 1: Loading alignment results from {alignment_json}...")
    with open(alignment_json) as f:
        alignment = json.load(f)

    # Decode DJI metadata (Default/Lens/Quaternion) for every camera, including
    # the reference, so each video has a matching *_metadata.json. The reference's
    # metadata (e.g. H1_metadata.json) is kept verbatim — it is the time origin;
    # non-reference metadata is trimmed in copy_metadata_with_trim() below.
    print("\nDecoding DJI metadata JSONs for all cameras (telemetry-parser)...")
    for cam_video in video_dict.values():
        ensure_camera_metadata(cam_video)

    # Get reference video info
    ref_video = video_dict[reference]
    ref_duration = get_video_duration(ref_video)
    ref_fps = get_video_fps(ref_video)
    ref_width, ref_height = get_video_resolution(ref_video)

    print(f"  Reference '{reference}': {ref_duration:.2f}s, {ref_fps:.2f}fps, {ref_width}x{ref_height}")

    output_videos = {reference: ref_video}

    # Process each non-reference video
    for camera_name in video_dict.keys():
        if camera_name == reference:
            continue

        src_video = video_dict[camera_name]
        delay_seconds = alignment[camera_name]['delay_seconds']

        # Generate output filename
        base_name = Path(src_video).stem
        ext = Path(src_video).suffix
        output_video = f"{base_name}{output_suffix}{ext}"

        print(f"\nStep 2: Processing '{camera_name}' (delay: {delay_seconds:+.4f}s)...")
        print(f"  Input: {src_video}")
        print(f"  Output: {output_video}")

        # Determine trim offset from the original timeline.
        # delay < 0  => camera is AHEAD of reference => drop leading frames.
        # delay >= 0 => camera is BEHIND (or equal); keep its real start. We do
        #               NOT prepend black frames because synthetic frames have no
        #               sensor/IMU data, so the output starts from real recording.
        if delay_seconds < 0:
            trim_offset = abs(delay_seconds)
        else:
            trim_offset = 0.0
            if delay_seconds > 0:
                print(f"  ⚠ '{camera_name}' is {delay_seconds:.3f}s BEHIND '{reference}'. "
                      f"Sensor/IMU data cannot be padded synthetically, so no black "
                      f"frames are prepended — output is shorter than the reference.")

        # Build ffmpeg command. ffmpeg's mp4 muxer CANNOT write DJI's djmd/dbgi
        # data streams ("Could not find tag for codec none in container"), so
        # they cannot be re-embedded into the aligned container by any ffmpeg
        # path (re-encode OR -c copy remux). Their full decoded content — lens
        # calibration + per-frame IMU/Quaternion samples — is instead preserved
        # via copy_metadata_with_trim() below, which is the authoritative
        # metadata carrier. Here we re-encode only the main video (frame-exact
        # trim) and audio, with NO rescale so each camera keeps its native
        # resolution and valid lens calibration.
        cmd = (
            f'ffmpeg -i "{src_video}" '
            f'-map 0:v:0 -map 0:a:0 '
            f'-ss {trim_offset:.4f} -t {ref_duration:.4f} '
            f'-c:v:0 libx265 -preset medium -crf 18 '
            f'-c:a:0 aac -b:a 128k '
            f'-copyts -avoid_negative_ts make_zero '
            f'-y "{output_video}" '
            f'-loglevel error'
        )

        # Execute ffmpeg
        print(f"  Running ffmpeg conversion (frame-exact trim, native resolution)...")
        result = subprocess.run(cmd, shell=True)

        if result.returncode == 0:
            print(f"  ✓ Successfully created {output_video}")
            output_videos[camera_name] = output_video
            # Copy + time-sync the DJI metadata JSON to the trimmed timeline.
            copy_metadata_with_trim(src_video, output_video,
                                    trim_offset, ref_duration)
        else:
            print(f"  ✗ Failed to create {output_video}")

    print("\n✓ Video synchronization complete!")
    print("\nAligned videos:")
    for name, video in output_videos.items():
        print(f"  {name}: {video}")

    return output_videos


def main():
    parser = argparse.ArgumentParser(
        description='Align audio tracks from multiple video files'
    )
    parser.add_argument(
        '--videos',
        type=str,
        help='JSON file with video mapping (default: {"H": "H.MOV", "L": "L.MP4", "R": "R.MP4"})'
    )
    parser.add_argument(
        '--reference',
        type=str,
        default='H',
        help='Reference camera name (default: H)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='audio_alignment.json',
        help='Output JSON file (default: audio_alignment.json)'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=16000,
        help='Sample rate for processing (default: 16000 Hz)'
    )
    parser.add_argument(
        '--sync-video',
        action='store_true',
        help='After audio alignment, synchronize videos and generate aligned output'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_aligned',
        help='Suffix for aligned video files (default: _aligned)'
    )

    args = parser.parse_args()

    # Default video mapping
    video_dict = {
        'H': 'H.MOV',      # Head camera
        'L': 'L.MP4',      # Left hand camera
        'R': 'R.MP4'       # Right hand camera
    }

    # Load custom video mapping if provided
    if args.videos:
        with open(args.videos) as f:
            video_dict = json.load(f)

    # Run audio alignment
    align_videos(video_dict, reference=args.reference, output_json=args.output, sr=args.sr)

    # Optionally sync videos
    if args.sync_video:
        print("\n" + "="*60)
        print("Starting video synchronization...")
        print("="*60 + "\n")
        sync_video_to_reference(args.output, video_dict, reference=args.reference,
                                output_suffix=args.output_suffix)


if __name__ == '__main__':
    main()
