"""
Full Instrument Segmentation using MedSAM2 with Point Prompts.

Instead of using the CholecT50 tip bounding boxes as box prompts (which only
segments the tip), this script uses the CENTER of each tip box as a positive
point prompt.  SAM2 then segments the entire connected object — capturing the
full instrument body (shaft + tip).

Usage
-----
# Quick pilot on 3 frames from VID68:
python segment_full_instrument.py

# Custom run:
python segment_full_instrument.py --videos VID68 VID70 --frames-per-video 5
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ---------- path setup so we can import from the medsam2 repo ----------
MEDSAM2_ROOT = Path(__file__).resolve().parent.parent / "medsam2"
sys.path.insert(0, str(MEDSAM2_ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ── colour palette per instrument id ──────────────────────────────────
INSTRUMENT_COLORS_BGR = {
    0: (128,  64, 128),   # grasper
    1: (244,  35, 232),   # bipolar
    2: ( 70,  70,  70),   # hook
    3: (102, 102, 156),   # scissors
    4: (190, 153, 153),   # clipper
    5: (153, 153, 153),   # irrigator
}


# ── CLI ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Segment full instruments with MedSAM2 point prompts."
    )
    p.add_argument(
        "--dataset-root", type=Path,
        default=Path(__file__).resolve().parent.parent / "cholect50-challenge-val",
        help="CholecT50 dataset root (contains labels/ and videos/).",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Where to save masks / overlays / summary.",
    )
    p.add_argument(
        "--videos", nargs="+",
        default=["VID68", "VID70", "VID73", "VID74", "VID75"],
        help="Video ids to process.",
    )
    p.add_argument(
        "--frames-per-video", type=int, default=9999,
        help="Max annotated frames to sample per video (9999 = all).",
    )
    p.add_argument(
        "--config", type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help="MedSAM2 config name.",
    )
    p.add_argument(
        "--checkpoint", type=Path,
        default=MEDSAM2_ROOT / "checkpoints" / "MedSAM2_latest.pt",
        help="MedSAM2 checkpoint path.",
    )
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────
def clamp_box(x1, y1, x2, y2, w, h):
    return max(0, min(w-1, x1)), max(0, min(h-1, y1)), \
           max(x1+1, min(w, x2)), max(y1+1, min(h, y2))


def normalized_xywh_to_xyxy(norm_box, width, height):
    x, y, bw, bh = norm_box
    x1 = int(round(x * width))
    y1 = int(round(y * height))
    x2 = int(round((x + bw) * width))
    y2 = int(round((y + bh) * height))
    return clamp_box(x1, y1, x2, y2, width, height)


def box_center(box_xyxy):
    """Return the (cx, cy) of a box in pixel coords."""
    x1, y1, x2, y2 = box_xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def sample_frames(annotation_dict, limit):
    frame_ids = sorted(int(fid) for fid in annotation_dict.keys())
    if len(frame_ids) <= limit:
        return frame_ids
    indices = np.linspace(0, len(frame_ids) - 1, num=limit, dtype=int)
    return [frame_ids[i] for i in indices]


def prepare_predictor(config_name, checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    model = build_sam2(config_name, str(checkpoint_path), device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor, device


def save_mask(mask, path):
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def overlay_mask(image_bgr, mask, color_bgr, alpha=0.45):
    overlay = image_bgr.copy()
    overlay[mask] = color_bgr
    return cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)


def draw_point(image, cx, cy, color_bgr, radius=6):
    """Draw a filled circle + cross-hair at the prompt point."""
    cv2.circle(image, (int(cx), int(cy)), radius, color_bgr, -1)
    cv2.circle(image, (int(cx), int(cy)), radius + 2, (255, 255, 255), 1)
    return image


def draw_box(image, box, color_bgr):
    """Draw the original tip box as a rectangle for reference."""
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)
    return image


def draw_label(image, box, label, color_bgr):
    """Draw the instrument name above the bounding box with a background."""
    x1, y1, x2, y2 = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    # Position label above the box; if too close to top edge, place inside
    label_y = y1 - 6
    if label_y - th < 0:
        label_y = y1 + th + 6
    label_x = x1
    # Draw background rectangle for readability
    cv2.rectangle(
        image,
        (label_x, label_y - th - 4),
        (label_x + tw + 4, label_y + 4),
        color_bgr, -1,
    )
    # Draw text in white
    cv2.putText(
        image, label, (label_x + 2, label_y),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
    return image


# ── main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root  = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    predictor, device = prepare_predictor(args.config, args.checkpoint.resolve())

    run_summary = {
        "method": "point_prompt_full_instrument",
        "dataset_root": str(dataset_root),
        "device": device,
        "config": args.config,
        "checkpoint": str(args.checkpoint.resolve()),
        "videos": {},
    }

    for video_id in args.videos:
        label_path = dataset_root / "labels" / f"{video_id}.json"
        video_dir  = dataset_root / "videos" / video_id
        if not label_path.exists() or not video_dir.exists():
            print(f"Skipping {video_id}: missing labels or frames.")
            continue

        data = json.loads(label_path.read_text(encoding="utf-8"))
        annotations   = data.get("annotations", {})
        categories    = data.get("categories", {})
        instrument_map = categories.get("instrument", {})
        target_map     = categories.get("target", {})

        selected = sample_frames(annotations, args.frames_per_video)
        vid_out  = output_root / video_id
        (vid_out / "overlays").mkdir(parents=True, exist_ok=True)
        (vid_out / "masks").mkdir(parents=True, exist_ok=True)

        frame_summaries = []

        for frame_id in selected:
            anns = annotations[str(frame_id)]
            fname = f"{frame_id:06d}.png"
            img_path = video_dir / fname
            if not img_path.exists():
                print(f"  Frame {fname} not found, skipping.")
                continue

            image_rgb = np.array(Image.open(img_path).convert("RGB"))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            h, w = image_rgb.shape[:2]

            predictor.set_image(image_rgb)
            overlay = image_bgr.copy()
            instance_summaries = []

            for ann_idx, ann in enumerate(anns):
                instrument_id = int(ann[1])
                target_id     = int(ann[8])

                # Original tip box
                tip_box = normalized_xywh_to_xyxy(
                    [float(v) for v in ann[3:7]], w, h
                )

                # ── Step 1: Box prompt to find the actual instrument pixels ──
                masks_box, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array(tip_box, dtype=np.float32)[None, :],
                    multimask_output=False,
                )
                tip_mask = masks_box[0] > 0

                # ── Step 2: Find a guaranteed solid point ──
                # Use distance transform to find the pixel deepest inside the tip mask
                if tip_mask.any():
                    dist_transform = cv2.distanceTransform(tip_mask.astype(np.uint8), cv2.DIST_L2, 5)
                    cy, cx = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
                else:
                    # Fallback to simple box center if box prediction somehow fails
                    cx, cy = box_center(tip_box)

                # ── Step 3: Point prompt segmentation for full instrument ──
                point_coords = np.array([[cx, cy]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)  # positive

                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=True,   # get 3 candidates
                )
                # Pick the mask with the HIGHEST score (usually the largest
                # coherent object that includes the point)
                best_idx = int(np.argmax(scores))
                mask = masks[best_idx] > 0

                color = INSTRUMENT_COLORS_BGR.get(instrument_id, (255, 255, 255))
                inst_name = instrument_map.get(
                    str(instrument_id), str(instrument_id)
                )
                overlay = overlay_mask(overlay, mask, color)
                overlay = draw_box(overlay, tip_box, color)
                overlay = draw_label(overlay, tip_box, inst_name, color)
                overlay = draw_point(overlay, cx, cy, color)

                mask_name = f"{frame_id:06d}_obj{ann_idx:02d}.png"
                save_mask(mask, vid_out / "masks" / mask_name)

                instance_summaries.append({
                    "mask_file": str((vid_out / "masks" / mask_name).resolve()),
                    "triplet_id": int(ann[0]),
                    "instrument_id": instrument_id,
                    "instrument_name": instrument_map.get(
                        str(instrument_id), str(instrument_id)
                    ),
                    "target_id": target_id,
                    "target_name": target_map.get(str(target_id), str(target_id)),
                    "tip_box_xyxy": list(map(int, tip_box)),
                    "prompt_point": [float(cx), float(cy)],
                    "score": float(scores[best_idx]),
                    "mask_area_px": int(mask.sum()),
                })

            overlay_path = vid_out / "overlays" / fname
            cv2.imwrite(str(overlay_path), overlay)
            frame_summaries.append({
                "frame_id": frame_id,
                "image_file": str(img_path.resolve()),
                "overlay_file": str(overlay_path.resolve()),
                "instances": instance_summaries,
            })
            print(
                f"  {video_id} frame {fname} -> "
                f"{len(instance_summaries)} instruments segmented"
            )

        run_summary["videos"][video_id] = {
            "frames_processed": len(frame_summaries),
            "frames": frame_summaries,
        }

    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"\nDone!  Results saved to {output_root}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
