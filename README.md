# S.A.R.V.I.S (Surgical Assistant Robotic Vision & Instrument Segmentation)

S.A.R.V.I.S is a repository dedicated to high-fidelity surgical instrument tracking and spatial segmentation in endoscopic videos. This project leverages the state-of-the-art **MedSAM2** foundation model to automatically segment both surgical targets and full medical instruments from noisy background tissue. 

## Key Features
* **Two-Step Instrument Segmentation**: A robust new inference pipeline built to circumvent dataset noise (like bright glare on tissue or oversized bounding box labels). It works by using a bounding box prompt to isolate the instrument's tip, computing a distance transform to find the solid center point of the tool, and then running MedSAM2 a second time using this high-confidence center point to accurately segment the entire instrument shaft and body without leaking into the background tissue.
* **MedSAM2 Integration**: Full support for tracking instruments through surgical video frames accurately. The foundational MedSAM2 architecture is fully integrated into the project.
* **CholecT50 Challenge Support**: Built to automatically iterate over and process validation frames from the CholecT50 endoscopic video challenge dataset.

## Repository Structure

* `full_instrument_segmentation/` - Contains our robust dual-prompting logic for full instrument segmentation.
    * `segment_full_instrument.py`: The primary tracking script. Generated masks are scored and saved as visual overlays with tool names printed above them.
* `medsam2/` - The MedSAM2 core library containing model architecture, inference loops, and checkpoint requirements.
* `reformat_labels.py` - Root level utility script for quickly correcting dataset labels.

## Getting Started

1. Ensure the Python virtual environment is active.
2. Download the MedSAM2 model weights and place them inside `medsam2/checkpoints/`.
3. Run the robust two-step segmentation script:
   ```bash
   cd full_instrument_segmentation
   python segment_full_instrument.py
   ```
4. Find the resulting segmented frames in your `full_instrument_segmentation/output/` directory!
