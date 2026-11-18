"""Statistics shared by experiment reports."""

import math


def standard_error(samples):
    return samples.std(ddof=1) / math.sqrt(len(samples))
