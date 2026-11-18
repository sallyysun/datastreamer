"""Write experiment reports and show details on request."""

import os
from pathlib import Path


def verbose_print(*values):
    if os.environ.get("VERBOSE") == "1":
        print(*values)


def write_report(path, lines):
    text = "\n".join(lines)
    Path(path).write_text(text + "\n")
    verbose_print(text)
