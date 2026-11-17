"""The Chapter 6 experiment configuration, in one place: every analysis and plotting script takes its families,
estimators, memories, scenarios, rates, switch regimes, seeds, horizon, labels and colours from here. The simulation
side (horizon, burn-in, change point, MFF memories, KLL clip) is src/estimators/evaluation.hpp and registry.hpp."""

# Simulation design (Table 6.2)
SEEDS = range(42, 102)           # 60 evaluation seeds
T, BURN, CHANGE = 5000, 500, 2500
CONSTANT_KNOB = 0.6              # dependence knob d; rho_S = (6/pi) asin(d/2)
DRIFT_FROM, DRIFT_SPAN = 0.2, 0.6   # matched drift path: d from 0.2 to 0.8 (span kept exact)
JUMP_KNOB = (0.2, 0.8)           # d before and after the jump at CHANGE
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
FAMILY_COLOUR = {"gaussian": "#9a9a9a", "t": "#C2185B", "clayton": "#8E6BB8", "gumbel": "#E3B505"}



def sfx(family):
    return "" if family == "gaussian" else "_" + family


def rho_s(d):
    """Spearman's rho of the dependence knob d (the Gaussian-equivalent correlation)."""
    import numpy as np
    return 6 / np.pi * np.arcsin(d / 2)


# Estimators (Chapter 4). The plotting colour table keys RMPL as "mle".
ESTIMATORS = ["rmpl", "xiao", "hermite"]
EST_LABEL = {"rmpl": "RMPL", "xiao": "Counting Matrix", "hermite": "Hermite Series"}
EST_LABEL2 = {"rmpl": "RMPL", "xiao": "Counting\nMatrix", "hermite": "Hermite\nSeries"}   # two-line axis labels
EST_COLOUR_KEY = {"rmpl": "mle", "xiao": "xiao", "hermite": "hermite"}
EST_MARKER = {"rmpl": "^", "xiao": "o", "hermite": "s"}

# Forgetting (Chapter 5)
MEMORIES = [39, 99, 199, 399, 1999]                        # fixed N_eff arms = MFF candidates
MEMORY_KEYS = [str(m) for m in MEMORIES] + ["nocum"]       # 6.2 columns, "nocum" = no forgetting
MEMORY_CMAP_POS = {39: 0.82, 99: 0.64, 199: 0.47, 399: 0.30, 1999: 0.10}   # position on the memory colour map
VFF_RATE = 0.02                                            # development choice ("vff_slow")
VFF_RATES = [0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.2]     # sensitivity grid (appendix)
STRATEGY_COLOUR = {"fixed": "#333333", "vff_slow": "#8E6BB8", "mff": "#C2185B"}

# Family switches (6.5): generator before -> after the change at CHANGE, at matched rho_S
SWITCHES = [("gauss2t", "gaussian", "t"), ("gauss2clayton", "gaussian", "clayton"), ("clayton2gumbel", "clayton", "gumbel")]

# 6.7 storage budgets (KiB)
BUDGETS = [16, 64, 256]
