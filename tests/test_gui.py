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
    assert notes[1] == 'gray=1,mask="fov.png"'           # quoted: paths may contain , or ;
    assert notes[2] == 'mask="fov.png"'


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
    for name in ("isic/rules.json", "hrf/rules_H01.json", "hrf/rules_G12.json", "hrf/rules_DR07.json"):
        rules = json.load(open(os.path.join(root, "case_studies", name), encoding="utf-8"))
        assert rules and all(r["metric"] in g.METRICS for r in rules)

def test_batch_csv_columns_follow_rule_order(g, qapp, tmp_path, monkeypatch):
    """With 11 rules an alphabetical sort would put R10/R11 before R2.
    The columns must come out as R1, R2, ..., R11, then total_score."""
    exp = tmp_path / "expected"; stu = tmp_path / "students" / "s1"
    exp.mkdir(); stu.mkdir(parents=True)
    cv2.imwrite(str(exp / "out.png"), disk()); cv2.imwrite(str(stu / "out.png"), disk(r=29))
    rules = [{"output": "out.png", "metric": "MSE", "max_score": 1.0, "tolerances": "1,10",
              "mode": "bucket", "aggregation": "mean", "notes": ""} for _ in range(11)]
    w = g.BatchWorker(str(tmp_path / "students"), str(exp), "", rules, None, "")
    got = {}
    w.finished.connect(lambda res: got.setdefault("res", res))   # through the Qt signal, as in the GUI
    w.run()
    monkeypatch.setattr(g.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    win = g.ImageGraderApp(); win.progress = None
    out = tmp_path / "batch.csv"
    win.on_batch_finished(got["res"], str(out))
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    rule_ids = [int(c.split("_")[0][1:]) for c in header[1:-1]]
    assert header[0] == "student" and header[-1] == "total_score"
    assert rule_ids == sorted(rule_ids) and rule_ids[0] == 1 and rule_ids[-1] == 11



def test_csv_columns_stay_in_rule_order_when_first_student_misses_a_file(g, qapp, tmp_path, monkeypatch):
    """The first student has no file for R1, so R1_resized first appears in a
    later row; it must still be written inside R1's columns."""
    exp = tmp_path / "expected"; exp.mkdir()
    cv2.imwrite(str(exp / "a.png"), disk()); cv2.imwrite(str(exp / "b.png"), disk())
    for name, files in (("s1", ["b.png"]), ("s2", ["a.png", "b.png"])):
        d = tmp_path / "students" / name; d.mkdir(parents=True)
        for fn in files:
            cv2.imwrite(str(d / fn), disk(r=29))
    rules = [{"output": fn, "metric": "MSE", "max_score": 1.0, "tolerances": "1,10",
              "mode": "bucket", "aggregation": "mean", "notes": ""} for fn in ("a.png", "b.png")]
    w = g.BatchWorker(str(tmp_path / "students"), str(exp), "", rules, None, "")
    got = {}
    w.finished.connect(lambda res: got.setdefault("res", res))
    w.run()
    monkeypatch.setattr(g.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    win = g.ImageGraderApp(); win.progress = None
    out = tmp_path / "batch.csv"
    win.on_batch_finished(got["res"], str(out))
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == ["student",
                      "R1_MSE_mask", "R1_MSE_student_path", "R1_MSE_expected_path", "R1_MSE_resized", "R1_MSE_val", "R1_MSE_score",
                      "R2_MSE_mask", "R2_MSE_student_path", "R2_MSE_expected_path", "R2_MSE_resized", "R2_MSE_val", "R2_MSE_score",
                      "total_score"]


def test_reset_all_clears_the_rules_text_box(g, qapp):
    win = g.ImageGraderApp()
    win.rules_text.setPlainText('[{"output": "a.png", "metric": "MSE"}]')
    win.on_reset_all()
    assert win.rules_text.toPlainText() == ""


def test_exported_rules_have_no_duplicate_expected_cout(g, qapp, tmp_path, monkeypatch):
    """The console text is stored once, in "tolerances", and survives export -> import."""
    src = tmp_path / "rules.json"
    src.write_text(json.dumps([
        {"output": "cout", "metric": "contains", "max_score": 1, "tolerances": "Area: 1520, Perimeter",
         "mode": "bucket", "notes": "file=stdout.txt"},
        {"output": "img_stud.png", "metric": "MSE", "max_score": 2, "tolerances": "1,10", "mode": "bucket"}]),
        encoding="utf-8")
    out = tmp_path / "exported.json"
    monkeypatch.setattr(g.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(g.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(src), "")))
    monkeypatch.setattr(g.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    win = g.ImageGraderApp(); win.on_import_json(); win.on_export_rules(); win.on_refresh_rules_text()
    for rules in (json.loads(out.read_text(encoding="utf-8")), json.loads(win.rules_text.toPlainText())):
        assert all("expected_cout" not in r for r in rules)
        assert rules[0]["tolerances"] == "Area: 1520, Perimeter"
    monkeypatch.setattr(g.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(out), "")))
    win2 = g.ImageGraderApp(); win2.on_import_json()
    assert win2.table.item(0, 3).text() == "Area: 1520, Perimeter"


def _set_rows(win, g, rows):
    win.table.setRowCount(0)
    for r, cells in enumerate(rows):
        win.table.insertRow(r)
        for c, v in enumerate(cells):
            win.table.setItem(r, c, g.QTableWidgetItem(v))


def test_single_eval_runs_image_rules_next_to_a_cout_rule(g, qapp, tmp_path, monkeypatch):
    _stub_dialogs(g, monkeypatch)
    exp = tmp_path / "exp.png"; stud = tmp_path / "s1" / "img.png"; stud.parent.mkdir()
    cv2.imwrite(str(exp), disk()); cv2.imwrite(str(stud), disk())
    (stud.parent / "stdout.txt").write_text("Area: 1\n", encoding="utf-8")
    win = g.ImageGraderApp()
    win.expected_image_path, win.student_image_path = str(exp), str(stud)
    _set_rows(win, g, [["img.png", "IoU", "2", "0.9", "bucket", "mean", ""],
                       ["cout", "exact", "1", "Area: 1", "bucket", "mean", ""]])
    win.on_run_single()
    assert win.table.item(0, 7).text() == "Value 1.0000 | Score 2.00"
    assert win.table.item(1, 7).text() == "cout: 1.0 | score 1.00"


def test_rule_files_and_text_are_checked_before_the_table_changes(g, qapp, tmp_path, monkeypatch):
    """Numeric tolerances import as text; a JSON object that is not a list of
    rules leaves the table unchanged; a non-numeric max score can be exported."""
    src = tmp_path / "rules.json"
    src.write_text(json.dumps([{"output": "a.png", "metric": "MSE", "max_score": 2, "tolerances": 5,
                                "mode": "bucket", "result": "Value 9 | Score 0"}]), encoding="utf-8")
    _stub_dialogs(g, monkeypatch, str(src))
    win = g.ImageGraderApp(); win.on_import_json()
    assert win.table.item(0, 3).text() == "5" and win.table.item(0, 7).text() == ""
    win.rules_text.setPlainText(json.dumps({"output": "b.png", "metric": "IoU"}))
    win.on_apply_rules_text()
    assert win.table.rowCount() == 1 and win.table.item(0, 0).text() == "a.png"
    win.table.setItem(0, 2, g.QTableWidgetItem("abc"))
    out = tmp_path / "out.json"
    monkeypatch.setattr(g.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    win.on_export_rules(); win.on_refresh_rules_text()
    assert json.loads(out.read_text(encoding="utf-8"))[0]["max_score"] == "abc"
    assert "result" not in json.loads(out.read_text(encoding="utf-8"))[0]
    assert json.loads(win.rules_text.toPlainText())[0]["max_score"] == "abc"


def test_cancel_stops_the_batch_without_a_report(g, qapp, tmp_path):
    S = tmp_path / "students"
    for i in range(30):
        d = S / f"s{i:02d}"; d.mkdir(parents=True); cv2.imwrite(str(d / "img.png"), disk())
    rule = {"output": "img.png", "metric": "SIZE", "max_score": 1, "tolerances": "0", "mode": "bucket",
            "notes": "100x100"}
    w = g.BatchWorker(str(S), "", "", [rule], None, "")
    got = []
    w.finished.connect(lambda res: got.append(res), g.Qt.DirectConnection)
    w.progress.connect(lambda *a: w.requestInterruption(), g.Qt.DirectConnection)
    w.start(); w.wait(10000)
    assert got == [None]


def test_student_picker_skips_hidden_and_zip_folders(g, qapp, tmp_path):
    for n in (".git", "__MACOSX", "alice"):
        (tmp_path / n).mkdir()
    win = g.ImageGraderApp(); win.student_dir = str(tmp_path); win.refresh_student_list()
    assert [win.student_picker.itemText(i) for i in range(win.student_picker.count())] == ["alice"]


def test_cancel_during_the_last_student_writes_no_report(g, qapp, tmp_path):
    d = tmp_path / "students" / "s1"; d.mkdir(parents=True); cv2.imwrite(str(d / "img.png"), disk())
    rules = [{"output": "img.png", "metric": "SIZE", "max_score": 1, "tolerances": "0", "mode": "bucket",
              "notes": "100x100"}] * 3
    w = g.BatchWorker(str(tmp_path / "students"), "", "", rules, None, "")
    got = []
    w.finished.connect(lambda res: got.append(res), g.Qt.DirectConnection)
    w.progress.connect(lambda *a: w.requestInterruption(), g.Qt.DirectConnection)
    w.start(); w.wait(10000)
    assert got == [None]


def test_no_deprecated_exec_calls(g):
    import inspect
    assert "exec_(" not in inspect.getsource(g)
