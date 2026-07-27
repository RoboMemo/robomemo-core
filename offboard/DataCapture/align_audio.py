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
    cmd = f'ffmpeg -i {video_file} -q:a 9 -n {audio_output} -loglevel error'
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
                   hop_length: int = 512, sr: int = 16000) -> dict:
    """
    Find time delay between two signals using normalized cross-correlation.

    Args:
        ref_envelope: Reference signal envelope
        test_envelope: Test signal envelope
        hop_length: Hop length in samples for mel-spectrogram frame
        sr: Sample rate

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

    # Find peak
    max_corr_idx = np.argmax(np.abs(correlation))
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

            direction = "ahead of" if result['delay_frames'] > 0 else "behind"
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
    width, height = map(int, result.stdout.strip().split(','))
    return width, height


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

        # Build ffmpeg command
        if delay_seconds < 0:
            # Camera is ahead: trim from the start
            trim_start = abs(delay_seconds)
            # Calculate trim duration to match reference length
            trim_duration = ref_duration
            cmd = (
                f'ffmpeg -i "{src_video}" '
                f'-ss {trim_start:.4f} -t {trim_duration:.4f} '
                f'-vf "scale={ref_width}:{ref_height}" '
                f'-c:v libx264 -preset fast -crf 23 '
                f'-c:a aac -b:a 128k '
                f'-y "{output_video}" '
                f'-loglevel error'
            )
        else:
            # Camera is behind: add black frames/silence at start
            # Calculate how many frames to prepend
            fps = get_video_fps(src_video)
            frames_to_prepend = int(delay_seconds * fps)
            frame_duration = 1.0 / fps

            # Create a black frame image file
            width, height = get_video_resolution(src_video)
            black_img = 'black_frame.png'

            # Use ffmpeg to generate black frame
            gen_black = (
                f'ffmpeg -f lavfi -i color=black:s={width}x{height}:d={delay_seconds:.4f} '
                f'-vf "scale={ref_width}:{ref_height}" '
                f'-y black_video.mp4 -loglevel error'
            )
            subprocess.run(gen_black, shell=True)

            # Concatenate black video + original video
            concat_file = 'concat_list.txt'
            with open(concat_file, 'w') as f:
                f.write("file 'black_video.mp4'\n")
                f.write(f"file '{src_video}'\n")

            cmd = (
                f'ffmpeg -f concat -safe 0 -i {concat_file} '
                f'-t {ref_duration:.4f} '
                f'-vf "scale={ref_width}:{ref_height}" '
                f'-c:v libx264 -preset fast -crf 23 '
                f'-c:a aac -b:a 128k '
                f'-y "{output_video}" '
                f'-loglevel error'
            )

        # Execute ffmpeg
        print(f"  Running ffmpeg conversion...")
        result = subprocess.run(cmd, shell=True)

        if result.returncode == 0:
            print(f"  ✓ Successfully created {output_video}")
            output_videos[camera_name] = output_video
        else:
            print(f"  ✗ Failed to create {output_video}")

        # Cleanup temporary files
        for tmp_file in ['black_video.mp4', 'concat_list.txt', 'black_frame.png']:
            if Path(tmp_file).exists():
                Path(tmp_file).unlink()

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
