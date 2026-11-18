"""Statistics shared by experiment reports."""

import math


def standard_error(samples, *, scale=1):
    return scale * samples.std(ddof=1) / math.sqrt(len(samples))


def paired_mean_se(a, b):
    differences = (a - b).dropna()
    return differences.mean(), standard_error(differences)


def win_loss_tie(mean, se):
    """Classify a loss difference using a two-SE threshold."""
    if mean < -2 * se:
        return "W"
    if mean > 2 * se:
        return "L"
    return "T"
