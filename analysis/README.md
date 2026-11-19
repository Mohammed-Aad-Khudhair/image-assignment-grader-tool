# Analyses

Scripts that measure how ImgGrader's results depend on its settings, and how fast it is. All scripts use ImgGrader's own code (`GraderApplicationCode.py`) and the data in `case_studies/` and `test_kit/`. Results are written to `results/`.

| Script | Question | Output |
| --- | --- | --- |
| `thresholds.py` | How much do totals and rankings change if every tolerance is 10 % or 20 % stricter or more lenient, or if continuous instead of bucket scoring is used? | `results/thresholds.csv` |
| `resizing.py` | What does the automatic resizing of a student image of the wrong size do to values and scores? How do other interpolation methods compare? | `results/resizing_case_studies.csv`, `results/resizing_interpolation.csv` |
| `realistic_tasks.py` | Do typical rubrics rank every correct solution of six course tasks above every typical mistake, before and after calibrating the tolerances? | `results/realistic_tasks.csv` |
| `aggregation.py` | How much do MSE, MAE and ΔE depend on the aggregation (mean, median, max, trim10)? | `results/aggregation.csv` |
| `performance.py` | How long does a batch of 30, 100 and 300 students take, and how much memory does it use? | `results/performance_<system>.csv` |

Run from the repository root (packages: those in `requirements.txt`, `scipy`, and `Pillow` for `realistic_tasks.py`):

```
python analysis/thresholds.py
python analysis/resizing.py
python analysis/realistic_tasks.py
python analysis/aggregation.py
python analysis/performance.py
```

All scripts except `performance.py` give the same files on every run. `performance.py` measures time and memory, which depend on the computer; it writes one file per operating system.

## thresholds.py

The four case-study gradings (ISIC, HRF H01, G12, DR07; 4 students each) are graded again with changed rules:

- **Tolerances ±10 % and ±20 %.** Lower-is-better metrics: `t → t·(1+d)`. Higher-is-better metrics bounded by 1 (IoU, SSIM, NCC, Contours, ContoursStrict): the allowed error `1 − t` is scaled, `t → 1 − (1 − t)·(1+d)`. Positive `d` is more lenient. The per-pixel threshold of N_DIFF_PIXELS (`thr:10`) is kept.
- **Continuous mode** for every rule.

For every variant the file lists each student's original and new total and Kendall's τ-b between the two rankings.

Results:
- With ±10 % tolerances, 3 (stricter) and 1 (more lenient) of the 16 student totals change; with ±20 %, 6 and 4. The largest change of a total is 1.67 of 8 points.
- No pair of students changes order in any variant. τ-b is 1 or 0.913; 0.913 means two students that had different totals now have the same total.
- Continuous mode changes 15 of 16 totals, by up to 4.82 points, but keeps every ranking (τ-b = 1). It is much more generous for metrics whose values lie close to 1 for every student, such as IoU on large lesions: ISIC student S4 gets 6.82 instead of 2 points.

## resizing.py

**Case studies.** Every student mask is saved at 75 %, 50 % and 25 % of the reference size (nearest-neighbour, so the mask stays binary) and graded by ImgGrader, which scales it back to the reference size with area interpolation (`resized` = 1 in the CSV). Values and scores are compared with the full-size grading.

Results: no rule score and no total increased. Lower-resolution submissions lost points: at 75 % 7 of 52 rule scores (4 of 16 totals) went down, at 50 % 9 (6 totals), at 25 % 22 (10 totals). The correct submissions lose most, because the coarser boundary no longer matches the reference.

**Interpolation methods.** ISIC students S2–S4 (IoU) and four test-kit T02 images (`perfect`, `noise_sigma5`, `jpeg_q20`, `blur_sigma1`; MSE) are reduced to 50 % and 25 % and scaled back with nearest, bilinear, bicubic and area interpolation. The file lists the metric and SSIM before and after, and the chi-square distance between the grey-level histograms of the original-size and the round-trip image, `χ² = ½ Σ (p − q)² / (p + q)` (0 = identical distributions, 1 = disjoint).

Results:
- When the reference size is an exact multiple of the student's size (all T02 cases), area interpolation gives the same image as nearest-neighbour: it repeats pixels and does not smooth. For the photographs MSE grows and SSIM falls with every method (e.g. the perfect T02 image at 50 %: MSE 0 → 218.8, SSIM 1 → 0.785). The smoothing methods (bilinear, bicubic) give lower MSE than area, so they would hide more of the lost detail.
- The histogram distances are small (χ² ≤ 0.021 in all cases; ≤ 0.008 for area interpolation): resizing hardly changes the distribution of grey levels, so the score change comes from the lost detail, not from a shift in intensities.

## realistic_tasks.py

Six typical course tasks on two photos (the ISIC lesion photo and the scikit-image `coffee` photo): grey-level conversion, Gaussian blur (σ = 2), Otsu threshold, morphological opening, a 30 % saturation increase in HSV, and the Sobel gradient magnitude. The instructor's reference is computed with OpenCV. Each task has one to three **correct** submissions made with other libraries or by hand (Pillow, scikit-image, SciPy, NumPy) and two **typical mistakes** (e.g. σ = 6 instead of 2, a median instead of a Gaussian filter, closing instead of opening, RGB weights applied to BGR data, 8-bit overflow, only the x-gradient). 24 submissions in total, graded with two rubrics:

- **initial**: first-guess tolerances an instructor might choose without testing;
- **calibrated**: for blur, opening and Sobel, the best band is set so that it still contains the worst correct submission, and the Sobel rules use a mask that ignores the 3-pixel image border (libraries treat borders differently). Calibration uses only the correct submissions.

Results: with the initial rubric, every correct submission scored higher than every mistake in 4 of the 6 tasks. In the blur task, σ = 6 and the median filter received full points (SSIM ≥ 0.98, PSNR ≥ 40 dB); in the opening task, closing scored as high as a correct solution made with scikit-image (IoU 0.9938 against 0.9985). Correct Sobel solutions received only 1 of 3 points because of border differences (NCC 0.78). With the calibrated rubric, all 6 tasks were separated, and the correct Sobel solutions received full points (NCC 1.0 inside the mask). In the grey-level task the mistakes still received 2.5 of 3 points: SSIM hardly changes when the channel weights are wrong, and only MAE separates them.

## aggregation.py

MSE, MAE and ΔE with mean, median, max and trim10 aggregation for the 8 students of test-kit task T02 and the 5 students of T03, and Kendall's τ-b between the ranking under each aggregation and under the mean.

Results: median and trim10 hide sparse errors (blur, JPEG) and keep uniform ones (noise, colour cast): MSE for `blur_sigma1` is 87.5 (mean), 4.67 (median), 19.5 (trim10) and 21 948 (max). Rankings change with the aggregation (τ-b 0.43–1.0 against the mean), most with max. The aggregation therefore has to be chosen for the error the task is about: mean for overall quality, median or trim10 to ignore small local defects, max to require that no pixel is far off.

## performance.py

Classes of 30, 100 and 300 students are built by copying the four case-study submissions and graded in Expected Dir mode. Each class size runs in its own process.

Results on the reference computer (Intel Xeon 2.30 GHz, 2 cores, Linux, Python 3.13): ISIC (4 rules, 1022 × 767 px) 0.03 s per student, 9.5 s for 300 students, about 146 MB; HRF H01 (3 rules, 3504 × 2336 px) 0.34 s per student, 103 s for 300 students, about 301 MB. Time grows linearly with the number of students and memory stays constant, because students are graded one after another.
