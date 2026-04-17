import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from segment_anything import sam_model_registry, SamPredictor


DATASET_ROOT = Path(r"c:\Users\Asus\Downloads\CholecT50-Challenge-Validation\cholect50-challenge-val")
MEDSAM2_ROOT = Path(r"c:\Users\Asus\Downloads\CholecT50-Challenge-Validation\medsam2")
MEDSAM2_SUMMARY = MEDSAM2_ROOT / "cholect50_pilot_output" / "run_summary.json"
SAM_CHECKPOINT = DATASET_ROOT / "sam_vit_b_01ec64.pth"
OUTPUT_DIR = MEDSAM2_ROOT / "benchmark_sam_vs_medsam2"


def load_medsam2_summary():
    return json.loads(MEDSAM2_SUMMARY.read_text(encoding="utf-8"))


def build_sam_predictor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)
    return SamPredictor(sam), device


def bbox_mask(shape, box):
    x1, y1, x2, y2 = box
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union else 0.0


def safe_div(num, den):
    return float(num / den) if den else 0.0


def segmentation_metrics(pred, ref):
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    tp = np.logical_and(pred, ref).sum()
    fp = np.logical_and(pred, np.logical_not(ref)).sum()
    fn = np.logical_and(np.logical_not(pred), ref).sum()
    union = np.logical_or(pred, ref).sum()
    pred_sum = pred.sum()
    ref_sum = ref.sum()
    dice = safe_div(2 * tp, pred_sum + ref_sum)
    iou = safe_div(tp, union)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "pred_area": int(pred_sum),
        "ref_area": int(ref_sum),
    }


def prompt_faithfulness(mask, prompt_box):
    box_mask = bbox_mask(mask.shape, prompt_box)
    inside = np.logical_and(mask, box_mask).sum()
    total = mask.sum()
    box_area = box_mask.sum()
    outside = np.logical_and(mask, np.logical_not(box_mask)).sum()
    mask_box = mask_bbox(mask)
    return {
        "mask_box_iou_to_prompt": bbox_iou(mask_box, prompt_box),
        "prompt_coverage": safe_div(inside, box_area),
        "inside_mask_ratio": safe_div(inside, total),
        "outside_box_ratio": safe_div(outside, total),
        "mask_area": int(total),
        "mask_box": mask_box,
    }


def mean_metric(rows, key):
    vals = [row[key] for row in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_medsam2_summary()
    predictor, device = build_sam_predictor()

    all_rows = []
    per_video_rows = {}

    for video_id, video_info in summary["videos"].items():
        rows = []
        for frame in video_info["frames"]:
            image_path = Path(frame["image_file"])
            image_rgb = np.array(Image.open(image_path).convert("RGB"))
            predictor.set_image(image_rgb)

            for inst in frame["instances"]:
                prompt_box = np.array(inst["prompt_box_xyxy"], dtype=np.float32)
                med_mask = np.array(Image.open(inst["mask_file"]).convert("L")) > 0
                sam_masks, sam_scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=prompt_box[None, :],
                    multimask_output=False,
                )
                sam_mask = sam_masks[0] > 0

                mask_metrics = segmentation_metrics(medsam2_mask := med_mask, sam_mask)
                med_prompt = prompt_faithfulness(medsam2_mask, inst["prompt_box_xyxy"])
                sam_prompt = prompt_faithfulness(sam_mask, inst["prompt_box_xyxy"])

                row = {
                    "video_id": video_id,
                    "frame_id": int(frame["frame_id"]),
                    "triplet_id": int(inst["triplet_id"]),
                    "instrument_id": int(inst["instrument_id"]),
                    "instrument_name": inst["instrument_name"],
                    "target_id": int(inst["target_id"]),
                    "target_name": inst["target_name"],
                    "sam_score": float(sam_scores[0]),
                    "medsam2_score": float(inst["score"]),
                    **mask_metrics,
                    "area_ratio_medsam2_to_sam": safe_div(mask_metrics["pred_area"], mask_metrics["ref_area"]),
                    "medsam2_mask_box_iou_to_prompt": med_prompt["mask_box_iou_to_prompt"],
                    "medsam2_prompt_coverage": med_prompt["prompt_coverage"],
                    "medsam2_inside_mask_ratio": med_prompt["inside_mask_ratio"],
                    "medsam2_outside_box_ratio": med_prompt["outside_box_ratio"],
                    "sam_mask_box_iou_to_prompt": sam_prompt["mask_box_iou_to_prompt"],
                    "sam_prompt_coverage": sam_prompt["prompt_coverage"],
                    "sam_inside_mask_ratio": sam_prompt["inside_mask_ratio"],
                    "sam_outside_box_ratio": sam_prompt["outside_box_ratio"],
                }
                rows.append(row)
                all_rows.append(row)

        per_video_rows[video_id] = rows

    def summarize(rows):
        return {
            "instances": len(rows),
            "mean_dice": mean_metric(rows, "dice"),
            "mean_iou": mean_metric(rows, "iou"),
            "mean_precision": mean_metric(rows, "precision"),
            "mean_recall": mean_metric(rows, "recall"),
            "mean_area_ratio_medsam2_to_sam": mean_metric(rows, "area_ratio_medsam2_to_sam"),
            "mean_medsam2_mask_box_iou_to_prompt": mean_metric(rows, "medsam2_mask_box_iou_to_prompt"),
            "mean_sam_mask_box_iou_to_prompt": mean_metric(rows, "sam_mask_box_iou_to_prompt"),
            "mean_medsam2_prompt_coverage": mean_metric(rows, "medsam2_prompt_coverage"),
            "mean_sam_prompt_coverage": mean_metric(rows, "sam_prompt_coverage"),
            "mean_medsam2_inside_mask_ratio": mean_metric(rows, "medsam2_inside_mask_ratio"),
            "mean_sam_inside_mask_ratio": mean_metric(rows, "sam_inside_mask_ratio"),
            "mean_medsam2_outside_box_ratio": mean_metric(rows, "medsam2_outside_box_ratio"),
            "mean_sam_outside_box_ratio": mean_metric(rows, "sam_outside_box_ratio"),
            "mean_medsam2_score": mean_metric(rows, "medsam2_score"),
            "mean_sam_score": mean_metric(rows, "sam_score"),
        }

    report = {
        "dataset_root": str(DATASET_ROOT),
        "device": device,
        "overall": summarize(all_rows),
        "per_video": {video_id: summarize(rows) for video_id, rows in per_video_rows.items()},
    }

    (OUTPUT_DIR / "benchmark_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "benchmark_rows.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved detailed rows to {OUTPUT_DIR / 'benchmark_rows.json'}")


if __name__ == "__main__":
    main()
