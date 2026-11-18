"""Shared study settings and Python presentation labels. Colours live in plot_theme.py."""
from decimal import Decimal
import json
from pathlib import Path

with (Path(__file__).resolve().parents[2] / "study_config.json").open() as file:
    STUDY = json.load(file)

# Simulation design (Table 6.2)
SEEDS = range(STUDY["first_seed"], STUDY["first_seed"] + STUDY["seed_count"])
T, BURN, CHANGE = (STUDY[key] for key in ("observations", "burn_in", "change_point"))
CONSTANT_RHO_S = STUDY["constant_rho_s"]                    # target Spearman rho of each scenario
DRIFT_FROM, DRIFT_TO = STUDY["drift_rho_s"]
DRIFT_SPAN = float(Decimal(str(DRIFT_TO)) - Decimal(str(DRIFT_FROM)))
JUMP_RHO_S = tuple(STUDY["jump_rho_s"])
SCENARIOS = ["constant", "drift", "jump"]
SCENARIO_LABEL = {"constant": "Constant", "drift": "Gradual drift", "jump": "Abrupt jump"}
WINDOWS = {"constant": ["whole"], "drift": ["whole"], "jump": ["whole", "pre", "rec_0_100", "rec_100_500", "post"]}

# Generating copula families. Results files carry a suffix: "" for Gaussian, "_<family>" otherwise.
FAMILIES = ["gaussian", "clayton", "gumbel", "t"]          # order of the 6.2 / 6.3 tables
FAMILIES_T31 = ["gaussian", "t", "clayton", "gumbel"]      # Table 3.1 order (6.4, 6.5, 6.7)
FAMILY_TEXT = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "t4"}           # plain text
FAMILY_MATH = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "$t_4$"}        # figures
FAMILY_LONG = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "Student's $t_4$"}
FAMILY_LONG_TEXT = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "Student's t4"}


def sfx(family):
    return "" if family == "gaussian" else "_" + family


# Estimators (Chapter 4)
ESTIMATORS = ["rmpl", "xiao", "hermite"]
EST_LABEL = {"rmpl": "RMPL", "xiao": "Counting Matrix", "hermite": "Hermite Series"}
EST_LABEL2 = {"rmpl": "RMPL", "xiao": "Counting\nMatrix", "hermite": "Hermite\nSeries"}   # two-line axis labels
EST_MARKER = {"rmpl": "^", "xiao": "o", "hermite": "s"}

# Forgetting (Chapter 5)
MEMORIES = STUDY["memories"]
MEMORY_KEYS = [str(m) for m in MEMORIES] + ["nocum"]       # 6.2 columns, "nocum" = no forgetting
VFF_RATE = STUDY["vff_rates"][0]
VFF_RATES = [0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.2]     # sensitivity grid (appendix)

# Family switches (6.5): generator before -> after the change at CHANGE, at matched rho_S
SWITCHES = [("gauss2t", "gaussian", "t"), ("gauss2clayton", "gaussian", "clayton"), ("clayton2gumbel", "clayton", "gumbel")]

# 6.7 storage budgets (KiB)
BUDGETS = [16, 64, 256]
