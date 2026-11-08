"""Shared fixtures. The application is a single file, so it is loaded by path.
Tests run headless (no window is shown)."""
import os
import sys
import importlib.util

import numpy as np
import cv2
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("grader", os.path.join(ROOT, "GraderApplicationCode.py"))
grader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grader)


@pytest.fixture(scope="session")
def g():
    return grader


@pytest.fixture(scope="session")
def qapp():
    return grader.QApplication.instance() or grader.QApplication(sys.argv)


def disk(h=100, w=100, r=30, hole=0, cx=None, cy=None):
    """White filled disk on black (BGR); optional black hole in the centre."""
    img = np.zeros((h, w, 3), np.uint8)
    c = (w // 2 if cx is None else cx, h // 2 if cy is None else cy)
    cv2.circle(img, c, r, (255, 255, 255), -1)
    if hole:
        cv2.circle(img, c, hole, (0, 0, 0), -1)
    return img


def run_batch(student_dir, rules, expected_dir="", expected_image="", mask_path=None, rules_base_dir=""):
    """Run the real BatchWorker synchronously; returns {student: row}."""
    w = grader.BatchWorker(student_dir, expected_dir, expected_image, rules, mask_path, rules_base_dir)
    out = {}
    w.finished.connect(lambda res: out.setdefault("res", res))
    w.run()
    return {r["student"]: r for r in out["res"]}
