# ImgGrader: Automated Grading for Image Processing Tasks

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE.txt)


## Description

ImgGrader is a Python-based GUI application for automated grading of image-processing assignments. It compares student outputs against instructor-provided reference images (and optional console output) using flexible, rule-based scoring. Instructors can configure grading rules through the graphical interface, no scripting required, making it easy to create objective, reproducible, and scalable grading workflows for large classes.

## Key Features

- **GUI-based rule configuration** – Define grading rules interactively (metrics, tolerances, scoring mode, aggregation, notes) without writing code.
- **Multiple image similarity metrics** – Support for EXACT, MSE/MAE/RMSE, PSNR, SSIM, NCC, IoU (Jaccard index), ∆E (CIEDE2000) in CIELAB, pixel-difference counts, object/hole miscounts and Euler-number difference (topology), contour and contour-hierarchy checks, and size checks.
- **Region of Interest (ROI) masking** – Load binary masks or draw ROIs to focus grading on clinically or pedagogically relevant regions. Each rule can have its own mask; with a mask, a metric uses only the masked pixels.
- **Console output evaluation** – Check text-based results (e.g., numeric outputs, messages) using exact, substring (“contains”), or regular expression matching.
- **Batch processing and CSV reports** – Evaluate all student submissions in a directory, with detailed CSV output including metrics, per-rule scores, and totals. For auditing, each rule also records the resolved expected and student file paths, the mask used, and whether the student image was resized (`resized`, area interpolation). Missing, ambiguous or failed evaluations are reported in the CSV and receive 0 points.
- **JSON-based rule import/export** – Save and reuse grading rules as JSON files; share rubrics across cohorts and semesters or version-control them.
- **Cross-platform & extensible** – Implemented in Python with PySide6, OpenCV, NumPy, and scikit-image; runs on Windows, macOS, and Linux.

## User Guide 

- Check App-Guide 

## Installation

### Prerequisites

- Python **3.8+**
- Git (for cloning the repository)
- A virtual environment is recommended (e.g., `venv` or `conda`).

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Mohammed-Aad-Khudhair/image-assignment-grader-tool.git

# 2. Change into the project directory
cd image-assignment-grader-tool

# 3. (Optional but recommended) Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start the application
python GraderApplicationCode.py
```

## Example rule files

The `rules/` folder contains importable rule sets for the two case studies in the paper:

- `isic_lesion_rules.json` – lesion segmentation (ISIC 2017, Task 1)
- `hrf_vessel_rules.json` – retinal vessel segmentation (HRF; images H01, G12, DR07)

Relative mask paths in a rule file (e.g. `mask=./expected/H01_fov.png`) are resolved from the folder of the rule file.

## Running the tests

```bash
pip install pytest
python -m pytest tests
```
