"""Regression test: `resources` must be imported before numpy in every entry point.

numpy/OpenBLAS reads OMP_NUM_THREADS and friends at IMPORT time. `resources` sets them. If an
auto-formatter (ruff's isort, in practice) reorders the import block and numpy lands first, the
thread cap silently stops working — the code still runs, the machine still becomes unusable.
That happened once; this test is why it cannot happen again.

The imports are fenced with `# isort: off` / `# isort: on`. This test guards the fence.
"""
import glob
import os
import re

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
NUMPY_RE = re.compile(r"^\s*(?:import numpy|from numpy)", re.M)
PANDAS_RE = re.compile(r"^\s*(?:import pandas|from pandas)", re.M)
SK_RE = re.compile(r"^\s*(?:import sklearn|from sklearn)", re.M)


def entry_points():
    out = []
    for p in sorted(glob.glob(os.path.join(SRC, "*.py"))):
        s = open(p, encoding="utf-8").read()
        if "import resources" in s:
            out.append((p, s))
    return out


def test_there_are_entry_points_to_check():
    assert entry_points(), "no guarded entry points found — did the guard get stripped?"


def test_resources_precedes_numpy_everywhere():
    for p, s in entry_points():
        r = s.find("import resources")
        for name, rx in (("numpy", NUMPY_RE), ("pandas", PANDAS_RE), ("sklearn", SK_RE)):
            m = rx.search(s)
            if m:
                assert r < m.start(), (
                    f"{os.path.basename(p)}: `import resources` (pos {r}) comes AFTER "
                    f"`{name}` (pos {m.start()}). The BLAS thread cap will not take effect."
                )


def test_isort_fence_is_present():
    """Without the fence, a formatter run silently reintroduces the bug."""
    for p, s in entry_points():
        assert "# isort: off" in s and "# isort: on" in s, (
            f"{os.path.basename(p)}: missing the isort fence around the resources import"
        )
