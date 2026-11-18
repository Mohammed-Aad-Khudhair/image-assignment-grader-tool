"""Batch runtime and peak memory for realistic class sizes.

A class of N students (N = 30, 100, 300) is built by copying the four
case-study submissions in turn, and graded with ImgGrader's batch evaluation
(Expected Dir mode, the case study's own rule file). Every class size runs in a
separate Python process, so the peak memory belongs to that run alone.

    python analysis/performance.py

Output: results/performance_<computer>.csv (one file per computer, so results
from different machines can be kept side by side).
"""
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (30, 100, 300)
CASE_NAMES = ("ISIC", "H01")


def peak_memory_mb():
    """Peak resident memory of this process in MB (Linux, macOS, Windows)."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    except ImportError:  # Windows
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return pmc.PeakWorkingSetSize / (1024 * 1024)


def cpu_name():
    if sys.platform.startswith("linux"):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def one_run(case_name, n):
    """Grade a class of n students; print one JSON line."""
    from _imggrader import CASES, g, load_rules, run_batch
    _, case_dir, rule_file = next(c for c in CASES if c[0] == case_name)
    rules = load_rules(os.path.join(case_dir, rule_file))
    out_name = rules[0]["output"]
    src = sorted(os.listdir(os.path.join(case_dir, "students")))
    tmp = tempfile.mkdtemp()
    try:
        for i in range(n):
            d = os.path.join(tmp, f"student{i:04d}")
            os.makedirs(d)
            shutil.copy(os.path.join(case_dir, "students", src[i % len(src)], out_name), os.path.join(d, out_name))
        t0 = time.perf_counter()
        rows = run_batch(tmp, os.path.join(case_dir, "expected"), rules, case_dir)
        seconds = time.perf_counter() - t0
    finally:
        shutil.rmtree(tmp)
    assert len(rows) == n and all(isinstance(r[f"R1_{rules[0]['metric']}_val"], float) for r in rows)
    h, w = g.imread(rows[0][f"R1_{rules[0]['metric']}_expected_path"]).shape[:2]
    print(json.dumps({"case": case_name, "students": n, "rules": len(rules), "image_size": f"{w}x{h}",
                      "seconds": round(seconds, 3), "seconds_per_student": round(seconds / n, 4),
                      "peak_memory_mb": round(peak_memory_mb(), 1)}))


def main():
    rows = []
    for case_name in CASE_NAMES:
        for n in SIZES:
            res = subprocess.run([sys.executable, os.path.abspath(__file__), case_name, str(n)],
                                 capture_output=True, text=True, check=True)
            r = json.loads(res.stdout.strip().splitlines()[-1])
            r.update({"cpu": cpu_name(), "logical_cores": os.cpu_count(),
                      "os": f"{platform.system()} {platform.release()}", "python": platform.python_version()})
            rows.append(r)
            print(f"{r['case']:5s} {n:4d} students: {r['seconds']:7.1f} s  "
                  f"({r['seconds_per_student']:.3f} s/student), peak memory {r['peak_memory_mb']:.0f} MB")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    name = f"performance_{platform.system().lower()}.csv"
    with open(os.path.join(HERE, "results", name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", os.path.join("analysis", "results", name), "|", rows[0]["cpu"], "|", rows[0]["os"])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        sys.path.insert(0, HERE)
        one_run(sys.argv[1], int(sys.argv[2]))
    else:
        main()
