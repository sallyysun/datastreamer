"""Write the section's comparison of fixed and adaptive strategies."""

from pathlib import Path
import sys

SECTION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SECTION_DIR.parent / "common"))
from strategy_allfixed import plot_all_fixed  # noqa: E402


if __name__ == "__main__":
    plot_all_fixed(SECTION_DIR)
