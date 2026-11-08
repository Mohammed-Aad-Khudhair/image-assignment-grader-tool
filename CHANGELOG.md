# Changelog

## 1.1

### Fixed – scoring and batch results
- Each rule has its own CSV columns (`R<n>_<metric>_...`), so two rules with the same metric no longer overwrite each other.
- Metric errors, NaN values and unknown metric names are reported as `ERROR` with 0 points instead of a silent value of 0.0 (which scored as perfect for error metrics such as MSE).
- A rule with empty or unreadable tolerances is reported as an error with 0 points instead of full points. EXACT without tolerances is pass/fail.
- Continuous scores are kept within [0, max score]; negative NCC or SSIM values no longer give negative points.
- Identical images (PSNR = infinity) receive full points instead of 0.
- Objects and Holes are miscounts against the reference (lower is better), as documented; Holes no longer counts the outer background.
- ContoursStrict counts objects that lie inside another object's hole.
- Batch CSV columns follow the rule order (R1, R2, …, total_score); they were previously sorted alphabetically.

### Fixed – files, masks and console output
- File matching: exact name first, then the same name without student suffixes (`_stud`, `_student`) and reference suffixes (`_gt`, `-reference`, `_ref`, `_expected`). Several candidates are reported as `AMBIGUOUS` with 0 points instead of picking one.
- Missing, unreadable and ambiguous student or reference files are reported separately in the CSV.
- File paths in the CSV are written in one normalised form (on Windows always with backslashes), whether they come from a file dialog, a rule file or the Expected Dir lookup.
- Importing rules no longer turns one rule's mask into a global mask, and loading a global mask no longer overwrites a rule's own `mask=`.
- Relative mask paths are read from the folder of the imported rule file; a missing mask is an error instead of an unmasked evaluation.
- The student image is resized to the reference size (area interpolation) before masking; this fixes a crash on size mismatch with a mask. The `resized` column records it.
- Console rules: a missing output file or an invalid regular expression is reported instead of a silent 0; expected text containing commas is no longer cut.
- A console rule without expected text is reported as an error with 0 points. Previously `contains` and `regex` rules with an empty text gave full points for any output (rules drafted by Scan Assignment Folder have no expected text until it is filled in).
- Images on paths with non-ASCII characters (e.g. accented folder names on Windows) are read and written correctly.
- The rectangle ROI drawing dialog is restored.

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
- `tests/`: 59 automated tests (`python -m pytest tests`).
- `rules/`: importable rule files for the ISIC lesion and HRF vessel examples (replaces the top-level `rules.json`, which was not valid JSON).
- `requirements.txt` lists the packages the code imports.
- `CITATION.cff` and this changelog.

### Documentation
- README, App-Guide, user guide PDF and licence file updated to match the program. The licence file now contains the unmodified GPL-3.0 text; the copyright notice is in the README.

### Effect on earlier results
On the ISIC and HRF case studies, all 52 rule scores are identical to the previous version. Four Objects values changed, because Objects is now a miscount against the reference instead of a raw count.
