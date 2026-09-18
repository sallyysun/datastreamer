"""Print the adaptive-memory tables from results/adaptive_memory_pairs.csv.

Usage: python3 show_adaptive.py [known_best|one_setting]
"""
import collections
import csv
import sys

SCEN = ["constant", "drift", "jump", "oscillate"]
FAM = ["xiao", "hermite", "hermite_db", "kalman"]
BASE = sys.argv[1] if len(sys.argv) > 1 else "known_best"

d = collections.defaultdict(dict)
for r in csv.DictReader(open("results/adaptive_memory_pairs.csv")):
    if r["baseline"] != BASE:
        continue
    d[(r["family"], r["scenario"])][r["method"]] = r

print("baseline:", BASE, "   (+sigma = adaptive is WORSE)\n")
for fam in FAM:
    print("===", fam)
    methods = sorted({m for s in SCEN for m in d[(fam, s)]})
    order = [BASE] + [m for m in methods if m != BASE]
    print("%-16s" % "", "".join("%24s" % s for s in SCEN))
    for m in order:
        line = "%-16s" % m
        for s in SCEN:
            r = d[(fam, s)].get(m)
            if not r:
                line += "%24s" % "-"
            elif m == BASE:
                line += "%24s" % ("%.4f  (mem %.0f)" % (float(r["rmse"]),
                                                        float(r["baseline_mem"])))
            else:
                line += "%24s" % ("%.4f  %+6.1f sig" % (float(r["rmse"]),
                                                        float(r["sigma"])))
        print(line)
    print()

sb = {}
for r in csv.DictReader(open("results/adaptive_memory.csv")):
    sb[(r["family"], r["method"])] = int(r["store_bytes"])
print("store_bytes")
for k in sorted(sb):
    if k[1] != "fixed":
        print("  %-12s %-16s %6d" % (k[0], k[1], sb[k]))
