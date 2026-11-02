# Challenge 1 ran in Colab on a GPU, so its results live in the saved outputs of
# the notebook. This reads them straight from there, so the page can't drift
# away from what was actually run.

import ast
import base64
import json
import os
import re
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOTEBOOK = os.path.join(ROOT, "Challenge1", "challenge1_prototype.ipynb")

ROW_RE = re.compile(r"^\d+\s+(\S+)\s+(normal|dim)\s+(.+?)\s+(\d)\s+([\d.]+)\s*$", re.M)
HOME_RE = re.compile(r"=== (\S+) \| task: (.+?)\nScene context: (\{.*?\n\})\n", re.S)


def _text(value):
    return "".join(value) if isinstance(value, list) else value


def _outputs(nb):
    for cell in nb["cells"]:
        yield from cell.get("outputs", [])


def _action(step):
    name, *args = step
    args = [round(a, 2) if isinstance(a, float) else a for a in args]
    return f"{name}({', '.join(str(a) for a in args)})"


def _homes(stream):
    homes = []
    found = list(HOME_RE.finditer(stream))
    for m, after in zip(found, found[1:] + [None]):
        rest = stream[m.end():after.start() if after else len(stream)]
        plan = re.search(r"^Plan: (.+)$", rest, re.M)
        actions = re.search(r"^Actions: (.+)$", rest, re.M)
        homes.append({"name": m.group(1), "task": m.group(2), "scene": json.loads(m.group(3)),
                      "plan": ast.literal_eval(plan.group(1)) if plan else [],
                      "actions": [_action(a) for a in ast.literal_eval(actions.group(1))] if actions else []})
    return homes


def _mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def _summary(rows):
    """The same grouping as the notebook's summary cell."""
    out = []
    for method in sorted({r["method"] for r in rows}):
        of_method = [r for r in rows if r["method"] == method]
        entry = {"method": method, "grasp": _mean(of_method, "grasp_success"), "iou": _mean(of_method, "iou")}
        for lighting in ("normal", "dim"):
            part = [r for r in of_method if r["lighting"] == lighting]
            entry[lighting] = {"grasp": _mean(part, "grasp_success"), "iou": _mean(part, "iou")}
        out.append(entry)
    return out


@lru_cache(maxsize=1)
def _load(mtime):
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)
    stream, table, figure = "", None, None
    for output in _outputs(nb):
        if output["output_type"] == "stream":
            stream += _text(output["text"])
        data = output.get("data", {})
        if "image/png" in data and figure is None:
            figure = base64.b64decode(_text(data["image/png"]))
        if "grasp_success" in _text(data.get("text/plain", "")) and table is None:
            table = _text(data["text/plain"])
    rows = [{"home": m[0], "lighting": m[1], "method": m[2], "grasp_success": int(m[3]), "iou": float(m[4])}
            for m in ROW_RE.findall(table or "")]
    return {"homes": _homes(stream), "rows": rows, "summary": _summary(rows)}, figure


def load():
    """(results, figure PNG bytes), reread only when the notebook changes."""
    return _load(os.path.getmtime(NOTEBOOK))
