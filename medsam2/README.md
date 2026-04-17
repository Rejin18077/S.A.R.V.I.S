# S.A.R.V.I.S.

S.A.R.V.I.S. is a lightweight research workspace for prompt-based surgical instrument segmentation on the CholecT50 validation set using MedSAM2 and SAM.

This repo contains the code used to:

- run tip-box prompted MedSAM2 segmentation on the labeled CholecT50 frames
- benchmark MedSAM2 against SAM on shared prompt boxes
- generate a synthetic 5 FPS sequence from the provided 1 FPS image folders

The repository intentionally excludes large datasets, generated masks, overlays, checkpoints, and synthetic frame outputs.

## Project Files

- `segment_cholect50_pilot.py`
  Runs MedSAM2 on CholecT50 labeled frames using the raw JSON tip boxes.

- `benchmark_sam_vs_medsam2.py`
  Re-runs SAM on the same prompt boxes and compares SAM vs MedSAM2 using overlap and prompt-faithfulness metrics.

- `make_synthetic_5fps.py`
  Builds a synthetic 5 FPS image sequence from the existing 1 FPS frame folders using optical-flow interpolation.

## Dataset Assumptions

Expected CholecT50 layout:

```text
cholect50-challenge-val/
  labels/
    VID68.json
    ...
  videos/
    VID68/
      000016.png
      ...
```

The JSON annotations provide bounding boxes and semantic metadata for sampled frames. They do not provide ground-truth segmentation masks.

## Setup

Create an environment and install the repo:

```bash
pip install -e .
```

For SAM benchmarking, place the SAM checkpoint at:

```text
cholect50-challenge-val/sam_vit_b_01ec64.pth
```

For MedSAM2 inference, place the MedSAM2 checkpoint at:

```text
checkpoints/MedSAM2_latest.pt
```

## Usage

Run MedSAM2 segmentation on labeled frames:

```bash
python segment_cholect50_pilot.py
```

Run SAM vs MedSAM2 benchmarking:

```bash
python benchmark_sam_vs_medsam2.py
```

Generate synthetic 5 FPS frame folders:

```bash
python make_synthetic_5fps.py
```

## Notes

- The MedSAM2 vs SAM benchmark compares predicted masks to each other and to the prompt boxes.
- It is not a ground-truth segmentation benchmark because the dataset only provides boxes, not pixel masks.
- Synthetic 5 FPS frames are interpolated from the available 1 FPS images, so they should be treated as pseudo-data rather than original captured video frames.

## Acknowledgments

This work builds on:

- [MedSAM2](https://github.com/bowang-lab/MedSAM2)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [CholecT50](https://github.com/CAMMA-public/cholect50)
