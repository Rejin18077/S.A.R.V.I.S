import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


INSTRUMENT_COLORS_BGR = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a small MedSAM2 pilot on CholecT50 frames using JSON box labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("..") / "cholect50-challenge-val",
        help="Path to the CholecT50 dataset root containing labels/ and videos/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cholect50_pilot_output"),
        help="Output folder inside the MedSAM2 workspace.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=["VID68", "VID70"],
        help="Video ids to sample from.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=3,
        help="Number of annotated frames to process per video.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help="MedSAM2 config name.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints") / "MedSAM2_latest.pt",
        help="MedSAM2 checkpoint path.",
    )
    return parser.parse_args()


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def normalized_xywh_to_xyxy(norm_box, width, height):
    x, y, w, h = norm_box
    x1 = int(round(x * width))
    y1 = int(round(y * height))
    x2 = int(round((x + w) * width))
    y2 = int(round((y + h) * height))
    return clamp_box(x1, y1, x2, y2, width, height)


def sample_frames(annotation_dict, limit):
    frame_ids = sorted(int(frame_id) for frame_id in annotation_dict.keys())
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


def save_mask(mask, out_path):
    mask_img = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(out_path), mask_img)


def overlay_mask(image_bgr, mask, color_bgr, alpha=0.45):
    overlay = image_bgr.copy()
    overlay[mask] = color_bgr
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return blended


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    predictor, device = prepare_predictor(args.config, args.checkpoint.resolve())
    run_summary = {
        "dataset_root": str(dataset_root),
        "device": device,
        "config": args.config,
        "checkpoint": str(args.checkpoint.resolve()),
        "videos": {},
    }

    for video_id in args.videos:
        label_path = dataset_root / "labels" / f"{video_id}.json"
        video_dir = dataset_root / "videos" / video_id
        if not label_path.exists() or not video_dir.exists():
            print(f"Skipping {video_id}: missing labels or frames.")
            continue

        with label_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", {})
        categories = data.get("categories", {})
        instrument_map = categories.get("instrument", {})
        target_map = categories.get("target", {})

        selected_frames = sample_frames(annotations, args.frames_per_video)
        video_output = output_root / video_id
        (video_output / "overlays").mkdir(parents=True, exist_ok=True)
        (video_output / "masks").mkdir(parents=True, exist_ok=True)
        frame_summaries = []

        for frame_id in selected_frames:
            anns = annotations[str(frame_id)]
            frame_name = f"{frame_id:06d}.png"
            image_path = video_dir / frame_name
            image_rgb = np.array(Image.open(image_path).convert("RGB"))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            height, width = image_rgb.shape[:2]

            predictor.set_image(image_rgb)
            overlay = image_bgr.copy()
            instance_summaries = []

            for ann_idx, ann in enumerate(anns):
                instrument_id = int(ann[1])
                verb_id = int(ann[7])
                target_id = int(ann[8])

                original_box = normalized_xywh_to_xyxy(
                    [float(v) for v in ann[3:7]],
                    width,
                    height,
                )
                prompt_box = original_box

                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array(prompt_box, dtype=np.float32)[None, :],
                    multimask_output=False,
                )
                mask = masks[0] > 0
                color_bgr = INSTRUMENT_COLORS_BGR.get(instrument_id, (255, 255, 255))
                overlay = overlay_mask(overlay, mask, color_bgr)
                cv2.rectangle(
                    overlay,
                    (prompt_box[0], prompt_box[1]),
                    (prompt_box[2], prompt_box[3]),
                    color_bgr,
                    2,
                )

                mask_name = f"{frame_id:06d}_obj{ann_idx:02d}.png"
                save_mask(mask, video_output / "masks" / mask_name)
                instance_summaries.append(
                    {
                        "mask_file": str((video_output / "masks" / mask_name).resolve()),
                        "triplet_id": int(ann[0]),
                        "instrument_id": instrument_id,
                        "instrument_name": instrument_map.get(str(instrument_id), str(instrument_id)),
                        "verb_id": verb_id,
                        "target_id": target_id,
                        "target_name": target_map.get(str(target_id), str(target_id)),
                        "original_box_xyxy": list(map(int, original_box)),
                        "prompt_box_xyxy": list(map(int, prompt_box)),
                        "score": float(scores[0]),
                    }
                )

            overlay_path = video_output / "overlays" / frame_name
            cv2.imwrite(str(overlay_path), overlay)
            frame_summaries.append(
                {
                    "frame_id": frame_id,
                    "image_file": str(image_path.resolve()),
                    "overlay_file": str(overlay_path.resolve()),
                    "instances": instance_summaries,
                }
            )
            print(f"Processed {video_id} frame {frame_name} with {len(instance_summaries)} instances.")

        run_summary["videos"][video_id] = {
            "frames_processed": len(frame_summaries),
            "frames": frame_summaries,
        }

    summary_path = output_root / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print(f"Saved pilot results to {output_root}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()
