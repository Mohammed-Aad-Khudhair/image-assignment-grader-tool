# ImgGrader test kit

A synthetic class for checking that ImgGrader grades correctly. It contains **9 grading tasks**, **69 students** and **459 student–rule results** whose correct value and score are known in advance. You grade the class with ImgGrader, then compare ImgGrader's CSV file with the known answers.

No real student data is used. Every student is constructed for one grading situation, for example "every pixel is 2 grey levels too bright" or "the file has the right name but is empty".

## Contents

| Path | Content |
| --- | --- |
| `T01_pixel_errors/` … `T09_rule_files/` | One folder per task (see below) |
| `check_my_csv.py` | Compares a CSV exported by ImgGrader with the correct answers of a task |
| `answer_key.json` | All 459 correct results in one file |
| `design.json` | Description of the tasks, used by the scripts in `generator/` |
| `generator/` | Scripts and source images that built the course and computed the answers |

Each task folder contains:

| Path | Content |
| --- | --- |
| `rules.json` | The rules to import into ImgGrader |
| `expected/` | The reference image(s) |
| `students/` | One subfolder per student |
| `masks/` | Masks used by the rules (T05, T06 only) |
| `EXPECTED_RESULTS.csv` | Correct value (or status) and score for every student and rule |
| `EXPECTED_TOTALS.csv` | Correct total score of every student |

## Quick use

1. Start ImgGrader and click **Reset All**.
2. **Import Rules (JSON)** → `T01_pixel_errors/rules.json`.
3. **Select Expected Dir** → `T01_pixel_errors/expected`; **Select Student Dir** → `T01_pixel_errors/students`.
4. **Batch Evaluate (student dir)** and save the CSV, e.g. as `T01_pixel_errors/my_result.csv`.
5. In a terminal, in this `test_kit` folder, run

   ```
   python check_my_csv.py T01_pixel_errors T01_pixel_errors/my_result.csv
   ```

   It prints every result that differs from the correct answer and ends with, for example,

   ```
   90 of 90 results match the expected results.
   ```

Repeat for the other tasks. Do not use **Load Mask / ROI** or **Load Expected Image** in these runs: masks are set by the rules themselves (T05, T06), and a loaded expected image would be used for every rule. T08 has no reference images: for T08, leave Expected Dir unset.

A result matches if the value differs from the correct value by at most 0.1 % (at most 0.001 for values below 1), or the status starts with the expected status (e.g. `MISSING student`), and the score differs by at most 0.2 % of the rule's maximum score.

## What each task checks

| Task | Students | Results | ImgGrader functions checked |
| --- | --- | --- | --- |
| **T01_pixel_errors** | 9 | 90 | MSE, MAE, RMSE, PSNR in bucket and continuous mode; EXACT; N_DIFF_PIXELS with `logic=any`, `logic=all` and `gray=1`; identical images (PSNR = ∞). Errors are constructed exactly: +1, +2, +5, +10, +20 on every channel, 37 changed pixels, one value off by 1. |
| **T02_perceptual** | 8 | 40 | SSIM, NCC and PSNR on blur, JPEG compression, noise, half contrast and an inverted image (negative NCC must give 0 points). |
| **T03_colour** | 5 | 15 | ΔE (CIEDE2000) with mean and max aggregation on a red colour cast, a 10° hue shift, a grey image and strong JPEG compression. |
| **T04_aggregation** | 3 | 36 | Aggregation `mean`, `median`, `max` and `trim10` for MAE, MSE and ΔE, with 5 %, 30 % and 60 % of the pixels changed. |
| **T05_masks_topology** | 12 | 96 | IoU, Objects, Holes, Euler, Contours, ContoursStrict on binary masks (eroded, dilated, shifted, extra objects, extra hole, island in a hole, split, inverted, empty, mask saved as 0/1, soft edges); N_DIFF_PIXELS inside a ring mask given with `mask=` in the rule. |
| **T06_roi** | 4 | 52 | Per-rule masks (left/right half, circle), masks at half resolution, and empty, missing or damaged masks, which must give ERROR and 0 points. |
| **T07_files_formats** | 17 | 51 | Finding student and reference files: reference named `*_gt.png`; student files named `_student`, `-stud`, saved as JPEG, two candidate files (`AMBIGUOUS student`), a similar but wrong name (`photo10_stud.png` → `MISSING student`), missing, damaged and empty files (`UNREADABLE student`); grey, RGBA and 16-bit images; wrong image size (resizing, SIZE); a file in a subfolder; Hungarian letters and spaces in folder names. |
| **T08_console** | 9 | 63 | Console rules `exact`, `contains`, `regex`: two-line text, extra spaces and empty lines, Windows line endings, an extra debug line, a wrong number, text with a comma, Hungarian text, empty output, missing output file, `file=log.txt`, and an invalid regular expression (ERROR). |
| **T09_rule_files** | 2 | 16 | Faulty or unusual rules: decimal max score (1.5), the same metric twice, an unknown metric name, empty or invalid tolerances and “—” (ERROR, 0 points), EXACT with “—” (pass/fail), an empty Output field (re-uses the file of the previous rule). |
| **Total** | **69** | **459** | |

## Single Eval spot checks

These five cases check **Run Single Eval** on screen. For each: **Reset All**, **Import Rules (JSON)** with the task's `rules.json`, **Load Expected Image** (the file in `expected/`), **Load Student Image** (the student's file), **Run Single Eval**. The **Result** column must show:

**1. T01, student `offset_plus2`** (+2 on every channel)

| Rule | 1 MSE | 2 MAE | 3 RMSE | 4 PSNR | 5 MSE cont. | 6 PSNR cont. | 7 EXACT | 8–10 N_DIFF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Value | 4.0000 | 2.0000 | 2.0000 | 42.1102 | 4.0000 | 42.1102 | 0.0000 | 0.0000 |
| Score | 3.00 | 3.00 | 3.00 | 4.00 | 1.92 | 1.75 | 0.00 | 3.00 |

**2. T05, student `island_in_hole`** (a hole with a small island inside: the Euler number does not change)

| Rule | 1 IoU | 2 IoU cont. | 3 Objects | 4 Holes | 5 Euler | 6 Contours | 7 ContoursStrict | 8 N_DIFF (ring) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Value | 0.9938 | 0.9938 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Score | 3.00 | 1.99 | 1.33 | 1.00 | 2.00 | 0.00 | 0.00 | 2.00 |

**3. T06, student `error_right_half`** (+20 on the right half only)

| Rule | Mask | Result |
| --- | --- | --- |
| 1 MSE | left | 0.0000 → 2.00 |
| 2 MSE | right | 400.0000 → 0.00 |
| 3 N_DIFF | left | 0.0000 → 2.00 |
| 4 N_DIFF | right | 136640.0000 → 0.00 |
| 5 MSE | left, half resolution | 0.0000 → 2.00 |
| 6 SSIM | circle | 0.9755 → 1.00 |
| 7 MSE | empty mask | ERROR (mask is empty) |
| 8 MSE | missing mask | ERROR (mask not found) |
| 9 MSE | none | 200.0000 → 1.00 |
| 10 EXACT | left | 1.0000 → 1.00 |
| 11 PSNR | right | 22.1102 → 0.00 |
| 12 N_DIFF | circle, half resolution | 22712.0000 → 0.00 |
| 13 MSE | damaged mask file | ERROR (cannot read mask) |

**4. T08, student `windows_line_endings`** — no images needed: **Select Student Dir** → `T08_console/students`, choose the student in **Current Student**, then **Run Single Eval**.

| Rule | 1 exact | 2 contains | 3 regex | 4 regex | 5 contains | 6 exact | 7 contains |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Result | 1.0 → 2.00 | 1.0 → 1.00 | 1.0 → 1.00 | ERROR: invalid regex | 0.0 → 0.00 | MISSING log.txt | 0.0 → 0.00 |

**5. T09, student `noisy`**

| Rule | 1 MSE | 2 MSE median | 3 SSIMM | 4 MSE | 5 IoU | 6 SIZE | 7 PSNR | 8 EXACT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Result | 64.1419 → 0.75 | 50.6667 → 0.75 | ERROR: unknown metric | ERROR: no tolerances set | ERROR: invalid tolerances | 0.0000 → 1.00 | ERROR: no tolerances set | 0.0000 → 0.00 |

## How the correct answers were obtained

The answers were **not** produced with ImgGrader. `generator/oracle.py` computes every value from the definitions in the ImgGrader documentation, with independent tools: OpenCV for reading, grey conversion and resizing, colour-science for CIEDE2000, SciPy for connected components, and SSIM written directly from its definition (Wang et al., 2004). Where a value is known by construction (e.g. +2 on every channel gives MSE = 4, a missing file gives `MISSING student`), the oracle is also checked against it: 183 such checks, all passed.

To rebuild the course and the answers (Python packages `numpy`, `opencv-python`, `scipy`, `colour-science`):

```
cd generator
python build_course.py sources ../rebuilt
python make_key.py ../rebuilt
python make_expected_csv.py ../rebuilt
```

The rebuilt course is identical to the one in this folder. `generator/run_suite.py` runs the whole kit through the real ImgGrader window automatically (Import Rules, Export/Import, Batch Evaluate, explicit expected image, Single Eval for every student) and compares 1856 checks with the answers:

```
python generator/run_suite.py ../GraderApplicationCode.py . results
```

## Image sources

All images are public domain or CC0. See `generator/sources/SOURCES.md`.
