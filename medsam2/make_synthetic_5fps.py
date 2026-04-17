import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create synthetic 5 FPS frame sequences from 1 FPS CholecT50 frames."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(r"c:\Users\Asus\Downloads\CholecT50-Challenge-Validation\cholect50-challenge-val\videos"),
        help="Folder containing the 1 FPS video frame directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"c:\Users\Asus\Downloads\CholecT50-Challenge-Validation\medsam2\videos_5fps_synthetic"),
        help="Folder where synthetic 5 FPS frame directories will be written.",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=["VID68", "VID70", "VID73", "VID74", "VID75"],
        help="Video ids to process.",
    )
    return parser.parse_args()


def load_frame(path):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read frame: {path}")
    return frame


def compute_flow(frame_a, frame_b):
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )


def warp_frame(frame, flow, t):
    h, w = frame.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0] * t).astype(np.float32)
    map_y = (grid_y + flow[..., 1] * t).astype(np.float32)
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def interpolate_pair(frame_a, frame_b):
    flow_ab = compute_flow(frame_a, frame_b)
    flow_ba = compute_flow(frame_b, frame_a)
    intermediates = []
    for step in range(1, 5):
        t = step / 5.0
        warped_a = warp_frame(frame_a, flow_ab, t)
        warped_b = warp_frame(frame_b, flow_ba, 1.0 - t)
        blend = cv2.addWeighted(warped_a, 1.0 - t, warped_b, t, 0)
        intermediates.append(blend)
    return intermediates


def process_video(video_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(video_dir.glob("*.png"))
    if len(frame_paths) < 2:
        return {"frames_in": len(frame_paths), "frames_out": 0}

    out_index = 0
    first = load_frame(frame_paths[0])
    cv2.imwrite(str(out_dir / f"{out_index:06d}.png"), first)
    out_index += 1

    for idx in range(len(frame_paths) - 1):
        frame_a = load_frame(frame_paths[idx])
        frame_b = load_frame(frame_paths[idx + 1])
        for inter in interpolate_pair(frame_a, frame_b):
            cv2.imwrite(str(out_dir / f"{out_index:06d}.png"), inter)
            out_index += 1
        cv2.imwrite(str(out_dir / f"{out_index:06d}.png"), frame_b)
        out_index += 1

    return {"frames_in": len(frame_paths), "frames_out": out_index}


def main():
    args = parse_args()
    summary = {}
    for video_id in args.videos:
        video_dir = args.input_root / video_id
        if not video_dir.exists():
            print(f"Skipping {video_id}: missing input frames.")
            continue
        out_dir = args.output_root / video_id
        stats = process_video(video_dir, out_dir)
        summary[video_id] = stats
        print(
            f"{video_id}: created {stats['frames_out']} synthetic frames from {stats['frames_in']} original 1 FPS frames."
        )

    summary_path = args.output_root / "summary.txt"
    args.output_root.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{video_id}: frames_in={stats['frames_in']}, frames_out={stats['frames_out']}"
        for video_id, stats in summary.items()
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
