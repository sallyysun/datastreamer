"""Print the adaptive-memory table from results/adaptive_memory_pairs.csv."""
import collections
import csv

SCEN = ["constant", "drift", "jump", "oscillate"]
FAM = ["xiao", "hermite", "hermite_db", "kalman"]

d = collections.defaultdict(dict)
for r in csv.DictReader(open("results/adaptive_memory_pairs.csv")):
    d[(r["family"], r["scenario"])][r["method"]] = r

for fam in FAM:
    print("===", fam)
    methods = sorted({m for s in SCEN for m in d[(fam, s)]})
    order = ["known_best"] + [m for m in methods if m != "known_best"]
    print("%-16s" % "", "".join("%24s" % s for s in SCEN))
    for m in order:
        line = "%-16s" % m
        for s in SCEN:
            r = d[(fam, s)].get(m)
            if not r:
                line += "%24s" % "-"
            elif m == "known_best":
                line += "%24s" % ("%.4f  (mem %.0f)" % (float(r["rmse"]),
                                                        float(r["known_best_mem"])))
            else:
                line += "%24s" % ("%.4f  %+6.1f sig" % (float(r["rmse"]),
                                                        float(r["sigma"])))
        print(line)
    print()
