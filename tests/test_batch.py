"""Batch evaluation: CSV columns, error handling, resizing, file matching, masks."""
import os

import cv2
import numpy as np
import pytest

from conftest import disk, run_batch


@pytest.fixture
def course(tmp_path):
    """expected/ with img1.png and img10.png, left/right half masks, and students."""
    E = tmp_path / "expected"; S = tmp_path / "students"; E.mkdir(); S.mkdir()
    cv2.imwrite(str(E / "img1.png"), disk())
    cv2.imwrite(str(E / "img10.png"), disk(r=10))
    left = np.zeros((100, 100), np.uint8); left[:, :50] = 255
    cv2.imwrite(str(E / "maskL.png"), left)
    cv2.imwrite(str(E / "maskR.png"), 255 - left)

    def student(name, files):
        d = S / name; d.mkdir()
        for fn, img in files.items():
            cv2.imwrite(str(d / fn), img)

    left_err = disk(); left_err[40:60, 20:40] = 0          # error only in the left half
    student("s_ok", {"img1_stud.png": disk()})
    student("s_left", {"img1_stud.png": left_err})
    student("s_small", {"img1_stud.png": cv2.resize(disk(), (50, 50), interpolation=cv2.INTER_AREA)})
    student("s_amb", {"img1_student.png": disk(), "img1-stud.png": disk()})
    student("s_sub", {"img10_stud.png": disk(r=10)})
    return tmp_path


def ndiff_rule(mask):
    return {"output": "img1_stud.png", "metric": "N_DIFF_PIXELS", "max_score": 1,
            "tolerances": "thr:10 | 0,100", "mode": "bucket", "notes": f"gray=1; mask={mask}"}


def test_two_rules_with_same_metric_get_separate_columns(course):
    E = course / "expected"
    res = run_batch(str(course / "students"), [ndiff_rule(E / "maskL.png"), ndiff_rule(E / "maskR.png")],
                    expected_dir=str(E))
    row = res["s_left"]
    assert row["R1_N_DIFF_PIXELS_val"] > 0          # left ROI sees the error
    assert row["R2_N_DIFF_PIXELS_val"] == 0         # right ROI does not


def test_errors_never_score_as_perfect(course):
    rules = [{"output": "img1_stud.png", "metric": "NOPE", "max_score": 1, "tolerances": "1", "mode": "bucket"}]
    row = run_batch(str(course / "students"), rules, expected_dir=str(course / "expected"))["s_ok"]
    assert str(row["R1_NOPE_val"]).startswith("ERROR") and row["R1_NOPE_score"] == 0.0


def test_missing_mask_is_an_error_not_unmasked(course):
    rule = ndiff_rule("does_not_exist.png")
    row = run_batch(str(course / "students"), [rule], expected_dir=str(course / "expected"))["s_ok"]
    assert str(row["R1_N_DIFF_PIXELS_val"]).startswith("ERROR")
    assert row["R1_N_DIFF_PIXELS_score"] == 0.0
    assert row["R1_N_DIFF_PIXELS_mask"].startswith("MISSING")


def test_relative_mask_resolves_from_rule_file_folder(course, monkeypatch):
    monkeypatch.chdir("/")                           # working directory must not matter
    rule = ndiff_rule("maskL.png")                   # relative to the rule file folder
    row = run_batch(str(course / "students"), [rule], expected_dir=str(course / "expected"),
                    rules_base_dir=str(course / "expected"))["s_left"]
    assert row["R1_N_DIFF_PIXELS_val"] > 0
    assert row["R1_N_DIFF_PIXELS_mask"].endswith("maskL.png")


def test_audit_columns_and_resizing(course):
    rules = [ndiff_rule(course / "expected" / "maskL.png"),
             {"output": "img1_stud.png", "metric": "SIZE", "max_score": 1, "tolerances": "0", "mode": "bucket"}]
    res = run_batch(str(course / "students"), rules, expected_dir=str(course / "expected"))
    ok, small = res["s_ok"], res["s_small"]
    assert ok["R1_N_DIFF_PIXELS_expected_path"].endswith("img1.png")
    assert ok["R1_N_DIFF_PIXELS_student_path"].endswith("img1_stud.png")
    assert ok["R1_N_DIFF_PIXELS_resized"] == 0
    assert small["R1_N_DIFF_PIXELS_resized"] == 1    # resized, and no crash with the mask
    assert isinstance(small["R1_N_DIFF_PIXELS_val"], float)
    assert small["R2_SIZE_val"] == 50.0 and small["R2_SIZE_resized"] == 0


def test_identical_psnr_full_points(course):
    rules = [{"output": "img1_stud.png", "metric": "PSNR", "max_score": 2, "tolerances": "30", "mode": "bucket"}]
    row = run_batch(str(course / "students"), rules, expected_dir=str(course / "expected"))["s_ok"]
    assert row["R1_PSNR_score"] == 2


def test_ambiguous_student_file_fails_fast(course):
    row = run_batch(str(course / "students"), [ndiff_rule(course / "expected" / "maskL.png")],
                    expected_dir=str(course / "expected"))["s_amb"]
    assert row["R1_N_DIFF_PIXELS_val"] == "AMBIGUOUS student"
    assert row["R1_N_DIFF_PIXELS_score"] == 0.0
    assert row["R1_N_DIFF_PIXELS_student_path"].startswith("AMBIGUOUS")


def test_no_substring_match(course):
    """img1 must not pick up img10_stud.png."""
    row = run_batch(str(course / "students"), [ndiff_rule(course / "expected" / "maskL.png")],
                    expected_dir=str(course / "expected"))["s_sub"]
    assert row["R1_N_DIFF_PIXELS_val"] == "MISSING student"


def test_console_text_with_commas(tmp_path, g):
    d = tmp_path / "students" / "s1"; d.mkdir(parents=True)
    (d / "stdout.txt").write_text("Area: 1, Perimeter: 2\n")
    rule = {"output": "cout", "metric": "exact", "max_score": 1,
            "tolerances": "Area: 1, Perimeter: 2", "mode": "bucket", "notes": "file=stdout.txt"}
    row = run_batch(str(tmp_path / "students"), [rule])["s1"]
    assert row["R1_exact_score"] == 1.0
    wrong = dict(rule, tolerances="Area: 1, Perimeter: 3")
    assert run_batch(str(tmp_path / "students"), [wrong])["s1"]["R1_exact_score"] == 0.0


def test_legacy_quoted_exp_cout_in_notes(g):
    assert g.expected_cout_text("", 'exp_cout="Area: 1, Perimeter: 2"; file=out.txt') == "Area: 1, Perimeter: 2"
    assert g.parse_notes_kv('exp_cout="a, b"; file=out.txt')["file"] == "out.txt"


def test_legacy_expected_cout_field_moves_to_tolerances(g):
    rule = {"output": "cout", "notes": "exp_cout=Area: 1, Perimeter: 2,file=stdout.txt",
            "expected_cout": "Area: 1, Perimeter: 2", "tolerances": ""}
    notes, tol = g._cout_fields(rule)
    assert tol == "Area: 1, Perimeter: 2" and notes == "file=stdout.txt"


def test_reference_suffixes_are_matched(tmp_path):
    """Real course layout: references named *_gt.png / *-reference.png,
    student files *_stud.png; expected images found via Expected Dir only."""
    E = tmp_path / "expected"; S = tmp_path / "students" / "s1"; E.mkdir(); S.mkdir(parents=True)
    cv2.imwrite(str(E / "H01_vessels_gt.png"), disk())
    cv2.imwrite(str(E / "H01_fov.png"), disk(r=45))              # other files in the folder are ignored
    cv2.imwrite(str(E / "lesion_mask-reference.png"), disk(r=20))
    cv2.imwrite(str(S / "H01_vessels_stud.png"), disk())
    cv2.imwrite(str(S / "lesion_mask_stud.png"), disk(r=20))
    rules = [{"output": "H01_vessels_stud.png", "metric": "IoU", "max_score": 1, "tolerances": "0.9", "mode": "bucket"},
             {"output": "lesion_mask_stud.png", "metric": "IoU", "max_score": 1, "tolerances": "0.9", "mode": "bucket"}]
    row = run_batch(str(tmp_path / "students"), rules, expected_dir=str(E))["s1"]
    assert row["R1_IoU_expected_path"].endswith("H01_vessels_gt.png") and row["R1_IoU_val"] == 1.0
    assert row["R2_IoU_expected_path"].endswith("lesion_mask-reference.png") and row["R2_IoU_val"] == 1.0


def test_reference_and_plain_file_both_present_is_ambiguous(tmp_path):
    E = tmp_path / "expected"; S = tmp_path / "students" / "s1"; E.mkdir(); S.mkdir(parents=True)
    cv2.imwrite(str(E / "img_gt.png"), disk()); cv2.imwrite(str(E / "img_ref.png"), disk())
    cv2.imwrite(str(S / "img_stud.png"), disk())
    rule = {"output": "img_stud.png", "metric": "IoU", "max_score": 1, "tolerances": "0.9", "mode": "bucket"}
    row = run_batch(str(tmp_path / "students"), [rule], expected_dir=str(E))["s1"]
    assert row["R1_IoU_val"] == "AMBIGUOUS expected" and row["R1_IoU_score"] == 0.0


def test_scanned_exact_rule_does_not_reward_wrong_image(course):
    """Regression: EXACT rule with empty tolerance gave full points to any image."""
    rule = {"output": "img1_stud.png", "metric": "EXACT", "max_score": 1, "tolerances": "", "mode": "bucket"}
    res = run_batch(str(course / "students"), [rule], expected_dir=str(course / "expected"))
    assert res["s_ok"]["R1_EXACT_score"] == 1.0
    assert res["s_left"]["R1_EXACT_score"] == 0.0


def test_rule_without_tolerances_is_reported(course):
    rule = {"output": "img1_stud.png", "metric": "MSE", "max_score": 1, "tolerances": "", "mode": "bucket"}
    row = run_batch(str(course / "students"), [rule], expected_dir=str(course / "expected"))["s_left"]
    assert str(row["R1_MSE_val"]).startswith("ERROR") and row["R1_MSE_score"] == 0.0


def test_unreadable_student_file_is_reported(tmp_path):
    E = tmp_path / "expected"; S = tmp_path / "students" / "s1"; E.mkdir(); S.mkdir(parents=True)
    cv2.imwrite(str(E / "img.png"), disk()); (S / "img_stud.png").write_text("not an image")
    rule = {"output": "img_stud.png", "metric": "IoU", "max_score": 1, "tolerances": "0.9", "mode": "bucket"}
    row = run_batch(str(tmp_path / "students"), [rule], expected_dir=str(E))["s1"]
    assert row["R1_IoU_val"] == "UNREADABLE student" and row["R1_IoU_score"] == 0.0


def test_console_missing_file_and_invalid_regex_are_reported(tmp_path):
    d = tmp_path / "students"; (d / "s1").mkdir(parents=True); (d / "s2").mkdir()
    (d / "s1" / "stdout.txt").write_text("algo=cleanup\n")
    rules = [{"output": "cout", "metric": "contains", "max_score": 1, "tolerances": "cleanup", "mode": "bucket", "notes": ""},
             {"output": "cout", "metric": "regex", "max_score": 1, "tolerances": "algo=(", "mode": "bucket", "notes": ""}]
    res = run_batch(str(d), rules)
    assert res["s1"]["R1_contains_score"] == 1.0
    assert res["s2"]["R1_contains_val"] == "MISSING stdout.txt" and res["s2"]["R1_contains_score"] == 0.0
    assert str(res["s1"]["R2_regex_val"]).startswith("ERROR: invalid regex") and res["s1"]["R2_regex_score"] == 0.0


def test_accented_student_folder_names(tmp_path, g):
    """Hungarian names (e.g. Kovács Ődön). Files are read via np.fromfile +
    cv2.imdecode, which also works with non-ASCII paths on Windows."""
    E = tmp_path / "expected"; S = tmp_path / "students" / "Kovács Ődön"; E.mkdir(); S.mkdir(parents=True)
    assert g.imwrite(str(E / "img.png"), disk()) and g.imwrite(str(S / "img_stud.png"), disk())
    rule = {"output": "img_stud.png", "metric": "IoU", "max_score": 1, "tolerances": "0.9", "mode": "bucket"}
    row = run_batch(str(tmp_path / "students"), [rule], expected_dir=str(E))["Kovács Ődön"]
    assert row["R1_IoU_val"] == 1.0 and row["R1_IoU_score"] == 1.0


def test_csv_paths_have_one_form_whatever_their_source(tmp_path):
    """The reference path is the same text whether the image was loaded with
    Load Expected Image (path as the file dialog returns it) or found in the
    Expected Dir; student and mask paths are normalised the same way."""
    E = tmp_path / "expected"; S = tmp_path / "students" / "s1"; E.mkdir(); S.mkdir(parents=True)
    cv2.imwrite(str(E / "img.png"), disk()); cv2.imwrite(str(S / "img_stud.png"), disk())
    m = np.full((100, 100), 255, np.uint8); cv2.imwrite(str(E / "m.png"), m)
    rule = {"output": "img_stud.png", "metric": "MSE", "max_score": 1, "tolerances": "1", "mode": "bucket",
            "notes": "mask=./expected/./m.png"}
    by_dir = run_batch(str(tmp_path / "students"), [rule], expected_dir=str(E), rules_base_dir=str(tmp_path))["s1"]
    explicit = run_batch(str(tmp_path / "students") + os.sep + ".", [rule], expected_dir=str(E),
                         expected_image=str(E) + os.sep + "." + os.sep + "img.png",
                         rules_base_dir=str(tmp_path))["s1"]
    for row in (by_dir, explicit):
        assert row["R1_MSE_expected_path"] == os.path.normpath(str(E / "img.png"))
        assert row["R1_MSE_student_path"] == os.path.normpath(str(S / "img_stud.png"))
        assert row["R1_MSE_mask"] == os.path.normpath(str(E / "m.png"))


@pytest.mark.parametrize("mode", ["exact", "contains", "regex"])
def test_console_rule_without_expected_text_is_an_error(g, tmp_path, mode):
    """An empty expected text is contained in every output and an empty regex
    matches anything; such a rule must give an error and 0 points, not full points."""
    S = tmp_path / "students" / "s1"; S.mkdir(parents=True)
    (S / "stdout.txt").write_text("totally wrong output\n", encoding="utf-8")
    val, err = g.console_rule(str(S), mode, "", "file=stdout.txt")
    assert val is None and err == "ERROR: no expected text set"
    rule = {"output": "cout", "metric": mode, "max_score": 1, "tolerances": "", "mode": "bucket",
            "notes": "file=stdout.txt"}
    row = run_batch(str(tmp_path / "students"), [rule])["s1"]
    assert row[f"R1_{mode}_score"] == 0.0 and row["total_score"] == 0.0
    assert row[f"R1_{mode}_val"] == "ERROR: no expected text set"


@pytest.mark.parametrize("field,value", [("mode", "bucketz"), ("aggregation", "avg"),
                                         ("max_score", "abc"), ("max_score", -2)])
def test_invalid_mode_aggregation_or_max_score_is_reported(course, field, value):
    """No silent defaults: an unknown mode or aggregation, or an unusable
    maximum score, is an ERROR with 0 points, not a score."""
    rule = {"output": "img1_stud.png", "metric": "MSE", "max_score": 2, "tolerances": "1,10",
            "mode": "bucket", "aggregation": "mean"}
    rule[field] = value
    row = run_batch(str(course / "students"), [rule], expected_dir=str(course / "expected"))["s_left"]
    assert str(row["R1_MSE_val"]).startswith("ERROR") and row["R1_MSE_score"] == 0.0


def test_mode_and_aggregation_ignore_case(course):
    base = {"output": "img1_stud.png", "metric": "MSE", "max_score": 2, "tolerances": "100,1000"}
    lower = run_batch(str(course / "students"), [dict(base, mode="bucket", aggregation="median")],
                      expected_dir=str(course / "expected"))["s_left"]
    upper = run_batch(str(course / "students"), [dict(base, mode="Bucket", aggregation="Median")],
                      expected_dir=str(course / "expected"))["s_left"]
    assert upper["R1_MSE_val"] == lower["R1_MSE_val"] and upper["R1_MSE_score"] == lower["R1_MSE_score"]


@pytest.mark.parametrize("layout", ["two_subfolders", "top_and_subfolder"])
def test_same_reference_name_in_two_places_is_ambiguous(tmp_path, layout):
    E = tmp_path / "expected"; S = tmp_path / "students" / "s1"; S.mkdir(parents=True)
    first = E / "a" if layout == "two_subfolders" else E
    for d in (first, E / "b"):
        d.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(d / "img.png"), disk())
    cv2.imwrite(str(S / "img.png"), disk())
    rule = {"output": "img.png", "metric": "IoU", "max_score": 1, "tolerances": "0.9", "mode": "bucket"}
    row = run_batch(str(tmp_path / "students"), [rule], expected_dir=str(E))["s1"]
    assert row["R1_IoU_val"] == "AMBIGUOUS expected" and row["R1_IoU_score"] == 0.0
    assert row["R1_IoU_expected_path"].startswith("AMBIGUOUS")


def test_unreadable_explicit_reference_is_reported_not_replaced(course):
    """An explicitly loaded reference that cannot be read must not be replaced
    silently by the reference found in the Expected Dir."""
    bad = course / "broken_reference.png"; bad.write_text("not an image")
    rule = {"output": "img1_stud.png", "metric": "IoU", "max_score": 1, "tolerances": "0.9", "mode": "bucket"}
    row = run_batch(str(course / "students"), [rule], expected_dir=str(course / "expected"),
                    expected_image=str(bad))["s_ok"]
    assert row["R1_IoU_val"] == "UNREADABLE expected" and row["R1_IoU_score"] == 0.0


def _one_student(tmp_path, files):
    d = tmp_path / "students" / "s1"; d.mkdir(parents=True)
    for name, data in files.items():
        p = d / name
        if isinstance(data, bytes):
            p.write_bytes(data)
        else:
            cv2.imwrite(str(p), data)
    return tmp_path / "students"


def test_reference_and_input_image_with_the_same_stem_are_ambiguous(tmp_path):
    """photo.png (the input) and photo_gt.png (the reference) both match
    photo_stud.png: the rule is AMBIGUOUS, on every OS and at any folder depth."""
    E = tmp_path / "expected"; E.mkdir()
    cv2.imwrite(str(E / "photo.png"), disk(r=10)); cv2.imwrite(str(E / "photo_gt.png"), disk())
    S = _one_student(tmp_path, {"photo_stud.png": disk()})
    rule = {"output": "photo_stud.png", "metric": "IoU", "max_score": 1, "tolerances": "0.9", "mode": "bucket"}
    row = run_batch(str(S), [rule], expected_dir=str(E))["s1"]
    assert row["R1_IoU_val"] == "AMBIGUOUS expected" and row["R1_IoU_score"] == 0.0


def test_reference_lookup_ignores_case(tmp_path):
    E = tmp_path / "expected"; E.mkdir(); cv2.imwrite(str(E / "Photo_GT.png"), disk())
    S = _one_student(tmp_path, {"photo_stud.png": disk()})
    rule = {"output": "photo_stud.png", "metric": "IoU", "max_score": 1, "tolerances": "0.9", "mode": "bucket"}
    row = run_batch(str(S), [rule], expected_dir=str(E))["s1"]
    assert row["R1_IoU_val"] == 1.0 and row["R1_IoU_score"] == 1.0


@pytest.mark.parametrize("data,val,status", [
    ("﻿Area: 100\n".encode("utf-8"), 1.0, None),              # UTF-8 with BOM
    ("Area: 100\r\n".encode("utf-16"), 1.0, None),                 # PowerShell > redirection
    ("Terület: 100\n".encode("cp1250"), None, "UNREADABLE stdout.txt"),
])
def test_console_output_encodings(tmp_path, data, val, status):
    S = _one_student(tmp_path, {"stdout.txt": data})
    rule = {"output": "cout", "metric": "regex", "max_score": 2, "tolerances": "^(Area|Terület): 100$", "mode": "bucket"}
    row = run_batch(str(S), [rule])["s1"]
    if status:
        assert str(row["R1_regex_val"]).startswith(status) and row["R1_regex_score"] == 0.0
    else:
        assert row["R1_regex_val"] == val and row["R1_regex_score"] == 2.0


def test_console_file_note_uses_the_same_parser_as_messages(tmp_path):
    S = _one_student(tmp_path, {"log.txt": b"Area: 100\n", "stdout.txt": b"wrong\n"})
    rule = {"output": "cout", "metric": "exact", "max_score": 1, "tolerances": "Area: 100",
            "mode": "bucket", "notes": "File=log.txt"}
    assert run_batch(str(S), [rule])["s1"]["R1_exact_score"] == 1.0
    rule["notes"] = 'file="my out.txt"'
    assert run_batch(str(S), [rule])["s1"]["R1_exact_val"] == "MISSING my out.txt"


def test_ncc_of_constant_images(tmp_path):
    """Identical constant images correlate perfectly; a constant image against
    a different one has no defined NCC and is an error, not 0 points."""
    flat = np.full((40, 40, 3), 200, np.uint8)
    E = tmp_path / "expected"; E.mkdir(); cv2.imwrite(str(E / "img.png"), flat)
    S = _one_student(tmp_path, {"img.png": flat})
    rule = {"output": "img.png", "metric": "NCC", "max_score": 1, "tolerances": "0.9", "mode": "bucket"}
    assert run_batch(str(S), [rule], expected_dir=str(E))["s1"]["R1_NCC_score"] == 1.0
    cv2.imwrite(str(S / "s1" / "img.png"), disk(h=40, w=40, r=10))
    row = run_batch(str(S), [rule], expected_dir=str(E))["s1"]
    assert str(row["R1_NCC_val"]).startswith("ERROR") and row["R1_NCC_score"] == 0.0


def test_hidden_and_zip_folders_are_not_students(tmp_path):
    S = _one_student(tmp_path, {"img.png": disk()})
    (S / "__MACOSX").mkdir(); (S / ".git").mkdir()
    rule = {"output": "img.png", "metric": "SIZE", "max_score": 1, "tolerances": "0", "mode": "bucket",
            "notes": "100x100"}
    assert set(run_batch(str(S), [rule])) == {"s1"}


def test_duplicate_student_files_are_listed_with_their_folders(tmp_path):
    S = _one_student(tmp_path, {"m_stud.png": disk()})
    (S / "s1" / "backup").mkdir(); cv2.imwrite(str(S / "s1" / "backup" / "m_stud.png"), disk())
    rule = {"output": "m_stud.png", "metric": "SIZE", "max_score": 1, "tolerances": "0", "mode": "bucket",
            "notes": "100x100"}
    row = run_batch(str(S), [rule])["s1"]
    assert row["R1_SIZE_val"] == "AMBIGUOUS student"
    assert os.path.join("backup", "m_stud.png") in row["R1_SIZE_student_path"]
