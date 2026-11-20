# Changelog

## 1.1

### Fixed – scoring and batch results
- Each rule has its own CSV columns (`R<n>_<metric>_...`), so two rules with the same metric no longer overwrite each other.
- Metric errors, NaN values and unknown metric names are reported as `ERROR` with 0 points instead of a silent value of 0.0 (which scored as perfect for error metrics such as MSE).
- A rule with empty or unreadable tolerances is reported as an error with 0 points instead of full points. EXACT without tolerances is pass/fail.
- Continuous scores are kept within [0, max score]; negative NCC or SSIM values no longer give negative points.
- Tolerances must be finite numbers of 0 or more; a negative, NaN or infinite tolerance is reported as `ERROR` with 0 points (a negative tolerance could give more than the max score). A tolerance written as a run of dashes (`---`) counts as empty.
- In continuous mode, a rule whose largest tolerance is 0 is pass/fail, as in bucket mode. Previously it always gave 0 points, also for a perfect result (Objects, Holes and Euler default to tolerance 0).
- Identical images (PSNR = infinity) receive full points instead of 0.
- Objects and Holes are miscounts against the reference (lower is better), as documented; Holes no longer counts the outer background.
- ContoursStrict counts objects that lie inside another object's hole.
- Batch CSV columns follow the rule order (R1, R2, …, total_score); they were previously sorted alphabetically. The order also holds when the first student has a missing file.
- NCC of a constant image is no longer 0: identical constant images give NCC = 1, and a constant image against a different one is reported as `ERROR` (NCC is undefined).
- An unknown scoring mode or aggregation, or a max score that is not a number of 0 or more, is reported as `ERROR` with 0 points. Previously an unknown mode was scored as continuous, an unknown aggregation as the mean, an unreadable max score as 1, and a negative max score gave negative points. Mode and aggregation names ignore case.

### Fixed – files, masks and console output
- File matching: exact name first, then the same name without student suffixes (`_stud`, `_student`) and reference suffixes (`_gt`, `-reference`, `_ref`, `_expected`). Several candidates are reported as `AMBIGUOUS` with 0 points instead of picking one; this includes reference files with the same name in different subfolders of the Expected Dir, and a reference that matches both by exact name and by stem (e.g. photo.png and photo_gt.png for photo_stud.png or photo.png). Reference lookup ignores case on every operating system. For students, candidates are listed with their subfolder.
- Missing, unreadable and ambiguous student or reference files are reported separately in the CSV. An explicitly loaded reference image that cannot be read is reported as `UNREADABLE expected` for every rule instead of being replaced by references from the Expected Dir.
- File paths in the CSV are written in one normalised form (on Windows always with backslashes), whether they come from a file dialog, a rule file or the Expected Dir lookup.
- Importing rules no longer turns one rule's mask into a global mask, and loading a global mask no longer overwrites a rule's own `mask=`.
- Relative mask paths are read from the folder of the imported rule file; a missing mask is an error instead of an unmasked evaluation.
- The student image is resized to the reference size (area interpolation) before masking; this fixes a crash on size mismatch with a mask. The `resized` column records it.
- An empty Output repeats the previous image rule with its status, so a missing file stays MISSING instead of an earlier file being graded; this also holds when the previous rule has invalid settings.
- SIZE without a reference reads only an explicit, positive size from the notes (WxH, size=WxH, expect_w/expect_h), never a number pair inside a file name; `thr=` must be a whole number, otherwise the rule is an `ERROR`.
- Console rules: a missing output file or an invalid regular expression is reported instead of a silent 0; expected text containing commas is no longer cut. Output files in UTF-8 with BOM or UTF-16 with BOM (PowerShell redirection) are read; other encodings, including UTF-16 without BOM, are reported as `UNREADABLE`, and the file name (not a folder name) ignores case, also in a subfolder (`file=out/log.txt`). `file=` and `mask=` in the notes are read with the same parser as the rest of the notes (case-insensitive keys, quoted values).
- A console rule without expected text is reported as an error with 0 points. Previously `contains` and `regex` rules with an empty text gave full points for any output (rules drafted by Scan Assignment Folder have no expected text until it is filled in).
- Images on paths with non-ASCII characters (e.g. accented folder names on Windows) are read and written correctly.
- The rectangle ROI drawing dialog is restored.
- Run Single Eval evaluates image rules also when the table contains a console rule.
- Cancel stops a batch evaluation (no CSV is written, also during the last student), and a second batch cannot be started while one is running.
- A loaded global mask is written into the notes in quotes, so its path may contain commas or semicolons; loading another mask replaces the previous one (previously the first mask stayed in use); rules added with Add Rule or drafted by Scan Assignment Folder keep their own `mask=` and console rules get no mask. `thr=`, `logic=` and `gray=` are read only as keys, not inside other keys or quoted values; the Current Student list skips `.git`/`__MACOSX` folders like the batch.
- Rule files and the rules text box are checked before the table is changed; numeric tolerances import as text; a non-numeric max score no longer breaks Export Rules or Refresh Text; exported rule files no longer contain old results.
- An unexpected error during a batch is reported instead of leaving the progress window open.
- `test_kit/check_my_csv.py` reports a status where a value is expected as `DIFFERENT` instead of stopping.
- Folders starting with `.` or `__` (e.g. `.git`, `__MACOSX`) are not treated as students; the image file dialogs accept BMP and TIFF.

### Changed
- Single and batch evaluation share one evaluation function. With a mask, every metric uses only the masked pixels.
- Max score accepts decimals (e.g. 1.5).
- New Euler metric: |χ(A) − χ(B)| with χ = objects − holes.
- CSV audit columns per rule: resolved student and reference paths, mask used, and whether the student image was resized.
- Window title is "ImgGrader".
- Reset All also clears the rules text box.
- Exported rule files store console text once, in `tolerances`; the duplicate `expected_cout` field is no longer written (older files that contain it still load).
- Removed a deprecated Qt call (`exec_`) that printed a warning when a batch started.

### Added
- `tests/`: 105 automated tests (`python -m pytest tests`).
- `case_studies/`: the ISIC lesion and HRF vessel case studies with images, masks, synthetic student submissions, rule files, exact results and the scripts that reproduce them (replaces the top-level `rules.json`, which was not valid JSON).
- `analysis/`: sensitivity of scores to tolerances, scoring mode, resizing and aggregation, and batch runtime and memory, with scripts and results.
- `test_kit/`: a synthetic class (9 tasks, 69 students, 459 results with independently computed answers) for checking that ImgGrader grades correctly.
- `requirements.txt` lists the packages the code imports.
- `CITATION.cff` and this changelog.

### Documentation
- README, App-Guide, user guide PDF and licence file updated to match the program. The licence file now contains the unmodified GPL-3.0 text; the copyright notice is in the README.

### Effect on earlier results
On the ISIC and HRF case studies, all 52 rule scores are identical to the previous version. Four Objects values changed, because Objects is now a miscount against the reference instead of a raw count.
