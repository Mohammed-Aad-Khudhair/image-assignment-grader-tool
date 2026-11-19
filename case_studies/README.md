# Case studies

The two case studies of the ImgGrader paper, with all files needed to grade them again: reference images, masks, student submissions, rule files, the scripts that made the submissions, and the exact results.

| Folder | Case study |
| --- | --- |
| `isic/` | Skin lesion segmentation: one ISIC 2017 image, 4 student masks, 4 rules |
| `hrf/` | Retinal vessel segmentation: three HRF images (healthy H01, glaucoma G12, diabetic retinopathy DR07), 4 students, 3 rules per image |
| `results/` | The CSV files written by ImgGrader for these cases (exact values) |
| `reproduce.py` | Grades both cases with ImgGrader and writes `results/` again |

The student submissions are **synthetic**: each was made from the reference mask by a fixed morphological change (see [How the submissions were made](#how-the-submissions-were-made)). No real student work is included.

## Grading the cases in ImgGrader

**ISIC**

1. **Reset All**, then **Import Rules (JSON)** → `isic/rules.json`.
2. **Select Expected Dir** → `isic/expected`; **Select Student Dir** → `isic/students`.
3. **Batch Evaluate (student dir)** and save the CSV.

**HRF**: the same steps with `hrf/rules_H01.json`, `hrf/expected` and `hrf/students`. Then import `hrf/rules_G12.json` and `hrf/rules_DR07.json` in turn and run the batch again; the two folders stay selected.

Do not use **Load Expected Image** or **Load Mask / ROI**: each rule finds its reference by file name in the Expected Dir (e.g. `H01_vessels_stud.png` → `H01_vessels_gt.png`), and the masks are given in the rules (`mask=`).

The totals must be:

| Student | ISIC | HRF H01 | HRF G12 | HRF DR07 |
| --- | --- | --- | --- | --- |
| S1 | 8.0 | 8.0 | 8.0 | 8.0 |
| S2 | 4.1667 | 3.1667 | 1.5 | 1.5 |
| S3 | 3.6667 | 3.9167 | 3.1667 | 3.1667 |
| S4 | 2.0 | 0.0 | 0.0 | 0.0 |

Every value and score per rule is in `results/` (`isic.csv`, `hrf_H01.csv`, `hrf_G12.csv`, `hrf_DR07.csv`), in the same format as ImgGrader's CSV export.

## Reproducing the results automatically

From the repository root:

```
python case_studies/reproduce.py
```

This runs ImgGrader's own batch evaluation on the four rule files and writes the four CSV files to `results/`. File paths in these files are relative to the case folder, so they are the same on every computer.

## The rules

**ISIC** (`isic/rules.json`, maximum 8 points)

| Rule | Metric | Max | Tolerances | Mask |
| --- | --- | --- | --- | --- |
| 1 | IoU | 5 | 0.995, 0.985, 0.975 | none |
| 2 | N_DIFF_PIXELS (`gray=1`) | 1 | thr:10 \| 4000, 8000 | `masks/roi_border.png`, a band around the lesion border |
| 3 | Objects | 1 | 1 | none |
| 4 | ContoursStrict | 1 | 1, 0.9 | none |

**HRF** (`hrf/rules_<ID>.json`, maximum 8 points per image)

| Rule | Metric | Max | Tolerances | Mask |
| --- | --- | --- | --- | --- |
| 1 | IoU | 5 | 0.80, 0.90, 0.95 (H01: 0.82, 0.90, 0.95) | field of view, `expected/<ID>_fov.png` |
| 2 | N_DIFF_PIXELS (`gray=1`) | 1.5 | thr:10 \| 5000, 30000 | band along the vessel centre lines, `expected/<ID>_centerline_ring.png` |
| 3 | N_DIFF_PIXELS (`gray=1`) | 1.5 | thr:10 \| 0, 5000 | band of background just outside the vessels, `expected/<ID>_bg_ring.png` |

## How the submissions were made

All scripts use only OpenCV and NumPy (`make_rings.py` also scikit-image). Run them from their own folder. They reproduce the files in this folder exactly; only the `stdout.txt` files of the HRF students differ, because they contain a random run time (they are not graded).

**ISIC** (`isic/`)

- `make_students.py` writes `students/S1` … `S4` from `expected/lesion_mask-reference.png`:
  - S1: the reference, smoothed (3×3 Gaussian) and thresholded again. It is identical to the reference.
  - S2: under-segmentation, erosion with a 3×3 elliptical element until IoU ≤ 0.986.
  - S3: over-segmentation, dilation until IoU ≤ 0.977.
  - S4: clearly wrong, shifted by (9, −6) pixels and eroded 3 times.
- `make_ring_and_overlays.py` writes the border band `masks/roi_border.png` (dilation minus erosion of the reference, 3×3 square element, 4 iterations, about 8 px wide) and overlay figures in `figures/`.

**HRF** (`hrf/`)

- `make_rings.py` writes, for every `<ID>_vessels_gt.png`, the centre-line band (skeleton dilated by radius 2, inside the field of view) and the background band (between 3 and 7 px from the vessels, inside the field of view).
- `make_students.py` writes `students/s1` … `s4` for every image:
  - s1: almost correct, closing, light smoothing and thresholding.
  - s2: under-segmentation, thin vessels removed by an opening followed by one erosion (3×3 elliptical element).
  - s3: over-segmentation, one dilation with the same element.
  - s4: clearly wrong, shifted by (10, −7) pixels and eroded twice with a 5×5 element.

## Sources and licences

| Files | Source | Licence |
| --- | --- | --- |
| `isic/expected/lesion.jpg`, `isic/expected/lesion_mask-reference.png` | ISIC 2017 Challenge, Task 1 training data, image `ISIC_0000000` and its ground-truth segmentation (Codella et al., 2018) | CC0 |
| `hrf/expected/<ID>_vessels_gt.png`, `hrf/expected/<ID>_fov.png` | High-Resolution Fundus (HRF) Image Database, images `01_h` (H01), `12_g` (G12), `07_dr` (DR07): manual vessel segmentation and field-of-view mask (Budai et al., 2013) | CC BY 4.0 |

All other files (masks, student submissions, results) were made from these with the scripts in this folder.

- Codella, N. C. F., et al. (2018). Skin lesion analysis toward melanoma detection: A challenge at the 2017 International Symposium on Biomedical Imaging (ISBI). *Proc. IEEE ISBI 2018*, 168–172.
- Budai, A., Bock, R., Maier, A., Hornegger, J., & Michelson, G. (2013). Robust vessel segmentation in fundus images. *International Journal of Biomedical Imaging*, 2013, 154860.
