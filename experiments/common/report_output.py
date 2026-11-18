"""Write experiment reports and show details on request."""

import os
from pathlib import Path


def verbose_print(*values):
    if os.environ.get("VERBOSE") == "1":
        print(*values)


def reports_enabled():
    return os.environ.get("REPORTS") == "1"


def write_report(path, lines):
    Path(path).write_text("\n".join(lines) + "\n")
    verbose_print(path)


def format_estimate(mean, uncertainty, mean_format="+.2f", uncertainty_format=".2f", *, separator=" ± "):
    return f"{mean:{mean_format}}{separator}{uncertainty:{uncertainty_format}}"


def md_table(columns, rows):
    return (["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
            + ["| " + " | ".join(r) + " |" for r in rows])
