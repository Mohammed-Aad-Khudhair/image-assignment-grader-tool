"""GUI behaviour that affects grading (run headless; dialogs are stubbed)."""
import json
import os

import cv2
import numpy as np

from conftest import disk, run_batch


def _stub_dialogs(g, monkeypatch, open_path=""):
    monkeypatch.setattr(g.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (open_path, "")))
    monkeypatch.setattr(g.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(g.QMessageBox, "warning", staticmethod(lambda *a, **k: None))


def test_import_does_not_turn_a_rule_mask_into_a_global_mask(g, qapp, tmp_path, monkeypatch):
    """Regression: importing rules used to make the first rule's mask global,
    so an IoU rule without a mask was computed inside that ROI."""
    (tmp_path / "masks").mkdir(); (tmp_path / "students" / "S2").mkdir(parents=True)
    gt = np.zeros((200, 200, 3), np.uint8); cv2.circle(gt, (100, 100), 60, (255, 255, 255), -1)
    st = np.zeros_like(gt); cv2.circle(st, (100, 100), 56, (255, 255, 255), -1)
    cv2.imwrite(str(tmp_path / "expected.png"), gt)
    cv2.imwrite(str(tmp_path / "students" / "S2" / "lesion_mask_stud.png"), st)
    G = (gt[:, :, 0] > 0).astype(np.uint8); k = np.ones((3, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "masks" / "ring.png"), (cv2.dilate(G, k, iterations=3) - cv2.erode(G, k, iterations=3)) * 255)
    rules = [{"output": "lesion_mask_stud.png", "metric": "IoU", "max_score": 5, "tolerances": "0.9", "mode": "bucket"},
             {"output": "lesion_mask_stud.png", "metric": "N_DIFF_PIXELS", "max_score": 1,
              "tolerances": "thr:10 | 4000,8000", "mode": "bucket", "notes": "gray=1; mask=./masks/ring.png"}]
    (tmp_path / "rules.json").write_text(json.dumps(rules))

    monkeypatch.chdir("/")                                  # app started from elsewhere
    _stub_dialogs(g, monkeypatch, str(tmp_path / "rules.json"))
    win = g.ImageGraderApp()
    win.on_import_json()
    assert win.mask_path == ""                              # no global mask taken from a rule
    assert win.rules_base_dir == str(tmp_path)

    table_rules = [{"output": win.table.item(r, 0).text(), "metric": win.table.item(r, 1).text(),
                    "max_score": float(win.table.item(r, 2).text()), "tolerances": win.table.item(r, 3).text(),
                    "mode": win.table.item(r, 4).text(), "aggregation": win.table.item(r, 5).text(),
                    "notes": win.table.item(r, 6).text()} for r in range(win.table.rowCount())]
    row = run_batch(str(tmp_path / "students"), table_rules, expected_image=str(tmp_path / "expected.png"),
                    mask_path=win.mask_path or None, rules_base_dir=win.rules_base_dir)["S2"]
    assert row["R1_IoU_val"] == g.metric_iou(gt, st)        # global IoU, not ring IoU
    assert row["R2_N_DIFF_PIXELS_mask"].endswith("ring.png")  # relative mask found via rule folder


def test_global_mask_does_not_overwrite_rule_masks(g, qapp):
    win = g.ImageGraderApp()
    for notes in ["gray=1; mask=cl_ring.png", "gray=1", ""]:
        r = win.table.rowCount(); win.table.insertRow(r)
        win.table.setItem(r, 0, g.QTableWidgetItem("x_stud.png"))
        win.table.setItem(r, 6, g.QTableWidgetItem(notes))
    g.ensure_mask_in_notes(win.table, "fov.png")
    notes = [win.table.item(r, 6).text() for r in range(3)]
    assert notes[0] == "gray=1; mask=cl_ring.png"           # own mask kept
    assert notes[1] == "gray=1,mask=fov.png"
    assert notes[2] == "mask=fov.png"


def test_rectangle_roi_is_scaled_to_full_image(g, qapp, tmp_path):
    """Restored ROI dialog: on a 1800x1200 image the preview is 900 px wide,
    so a rectangle drawn on the preview must be scaled by 2."""
    p = tmp_path / "big.png"; cv2.imwrite(str(p), np.zeros((1200, 1800, 3), np.uint8))
    dlg = g.RectRoiDialog(str(p))
    dlg.start, dlg.end = g.QRect(100, 50, 1, 1).topLeft(), g.QRect(300, 150, 1, 1).topLeft()
    assert dlg.rect_in_image() == (200, 100, 600, 300)


def test_rule_popup_accepts_half_points(g, qapp):
    popup = g.RulePopup()
    popup.max_score.setValue(1.5)
    assert popup.get_rule()["max_score"] == 1.5


def test_rule_popup_stores_console_text_in_tolerances(g, qapp):
    popup = g.RulePopup(autofill_output="cout")
    popup.cout_expected_edit.setText("Area: 1, Perimeter: 2")
    rule = popup.get_rule()
    assert rule["tolerances"] == "Area: 1, Perimeter: 2"
    assert "exp_cout" not in rule["notes"]


def test_example_rule_files_are_valid(g):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("isic_lesion_rules.json", "hrf_H01_rules.json", "hrf_G12_rules.json", "hrf_DR07_rules.json"):
        rules = json.load(open(os.path.join(root, "rules", name), encoding="utf-8"))
        assert rules and all(r["metric"] in g.METRICS for r in rules)
