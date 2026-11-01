"""Metric definitions, ROI handling and scoring."""
import math

import cv2
import numpy as np
import pytest

from conftest import disk


def test_identical_images_give_ideal_values(g):
    a = disk()
    assert g.compute_metric("MSE", a, a)[0] == 0
    assert math.isinf(g.compute_metric("PSNR", a, a)[0])
    assert g.compute_metric("SSIM", a, a)[0] == pytest.approx(1.0)
    assert g.compute_metric("NCC", a, a)[0] == pytest.approx(1.0)
    assert g.compute_metric("IoU", a, a)[0] == 1.0
    assert g.compute_metric("EXACT", a, a)[0] == 1.0
    assert g.compute_metric("N_DIFF_PIXELS", a, a, tols="thr:10 | 0,100")[0] == 0


@pytest.mark.parametrize("metric", ["MSE", "MAE", "RMSE", "PSNR", "SSIM", "NCC", "IoU", "EXACT"])
def test_unmasked_values_match_original_metric_functions(g, metric):
    """Without an ROI, the shared evaluation reproduces the original metric functions."""
    a, b = disk(r=30), disk(r=27, cx=52)
    assert g.compute_metric(metric, a, b)[0] == pytest.approx(g.METRICS[metric](a, b))


def test_unmasked_n_diff_matches_original(g):
    a, b = disk(r=30), disk(r=27)
    val, tol = g.compute_metric("N_DIFF_PIXELS", a, b, tols="thr:10 | 0,100", notes="gray=1")
    assert val == g.metric_n_diff_pixels(a, b, thr=10, use_gray=True)
    assert tol == "0,100"


def test_holes_do_not_count_outer_background(g):
    assert g._count_holes(disk()) == 0
    assert g._count_holes(disk(hole=5)) == 1


def test_objects_and_holes_are_miscounts_against_reference(g):
    a, b = disk(), disk(hole=5)
    assert g.metric_count_holes(a, b) == 1
    assert g.metric_count_objects(a, a) == 0


def test_euler_detects_topology_error_that_iou_misses(g):
    a, b = disk(), disk(hole=5)
    assert g.metric_iou(a, b) > 0.97
    assert g.metric_euler(a, b) == 1


def test_roi_metrics_use_roi_pixels_only(g):
    """An error outside the ROI must not change the value; inside the ROI the
    mean is taken over ROI pixels, not diluted by the rest of the image."""
    a = np.full((100, 100, 3), 100, np.uint8)
    b = a.copy()
    b[:, 50:] = 140                       # error only in the right half
    left = np.zeros((100, 100), bool); left[:, :50] = True
    right = ~left
    assert g.compute_metric("MSE", a, b, left)[0] == 0
    assert g.compute_metric("MSE", a, b, right)[0] == pytest.approx(40 ** 2)
    assert g.compute_metric("MSE", a, b)[0] == pytest.approx(40 ** 2 / 2)   # whole image
    assert g.compute_metric("N_DIFF_PIXELS", a, b, left, tols="thr:10|0")[0] == 0
    assert g.compute_metric("N_DIFF_PIXELS", a, b, right, tols="thr:10|0")[0] == 50 * 100


def test_size_metric_compares_original_dimensions(g):
    assert g.compute_metric("SIZE", np.zeros((100, 100, 3), np.uint8), np.zeros((50, 60, 3), np.uint8))[0] == 50


def test_unknown_metric_raises(g):
    with pytest.raises(KeyError):
        g.compute_metric("NOPE", disk(), disk())


@pytest.mark.parametrize("value,points", [(0.97, 5.0), (0.92, 10 / 3), (0.85, 5 / 3), (0.5, 0.0)])
def test_bucket_scoring_matches_user_guide_example(g, value, points):
    """User Guide, Table 2: IoU tolerances 0.82,0.90,0.95 -> 0, 1.67, 3.33, 5 points."""
    assert g.compute_score("IoU", value, 5, "0.82,0.90,0.95", "bucket") == pytest.approx(points)


def test_lower_is_better_bucket_scoring(g):
    assert g.compute_score("N_DIFF_PIXELS", 3000, 1, "4000,8000", "bucket") == 1.0
    assert g.compute_score("N_DIFF_PIXELS", 6710, 1, "4000,8000", "bucket") == 0.5
    assert g.compute_score("N_DIFF_PIXELS", 9374, 1, "4000,8000", "bucket") == 0.0


def test_continuous_scoring(g):
    assert g.compute_score("SSIM", 0.475, 2, "0.95", "continuous") == pytest.approx(1.0)
    assert g.compute_score("MSE", 50, 2, "100", "continuous") == pytest.approx(1.0)


def test_psnr_infinite_scores_full_points(g):
    assert g.compute_score("PSNR", float("inf"), 3, "30,25", "bucket") == 3


def test_empty_or_invalid_tolerances_never_award_points(g):
    """Regression: an empty or unreadable tolerance used to give full points."""
    with pytest.raises(ValueError):
        g.compute_score("MSE", 1e6, 2, "", "bucket")
    with pytest.raises(ValueError):
        g.compute_score("IoU", 0.1, 5, "0.9;0.8", "bucket")


def test_exact_without_tolerance_is_pass_fail(g):
    """EXACT rules created by 'Scan Assignment Folder' have no tolerance."""
    assert g.compute_score("EXACT", 1.0, 1, "", "bucket") == 1.0
    assert g.compute_score("EXACT", 0.0, 1, "", "bucket") == 0.0
    assert g.compute_score("EXACT", 0.0, 1, "—", "bucket") == 0.0   # App-Guide writes "—"


def test_continuous_score_never_negative(g):
    assert g.compute_score("NCC", -0.5, 2, "0.95", "continuous") == 0.0
    assert g.compute_score("SSIM", -0.2, 2, "0.9", "continuous") == 0.0


def test_contours_strict_counts_objects_inside_holes(g):
    """Regression: an object lying inside another object's hole was ignored."""
    ring_with_island = disk(r=40, hole=25)
    cv2.circle(ring_with_island, (50, 50), 8, (255, 255, 255), -1)      # island inside the hole
    ring_only = disk(r=40, hole=25)
    assert g._holes_signature(cv2.cvtColor(ring_with_island, cv2.COLOR_BGR2GRAY)) == [0, 1]
    assert g.metric_contours_strict(ring_with_island, ring_only) == 0.5
