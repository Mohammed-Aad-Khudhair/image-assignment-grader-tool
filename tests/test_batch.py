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
