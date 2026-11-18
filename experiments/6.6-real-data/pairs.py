"""The 32 pre-specified pairs of Section 6.6."""
PAIRS = [("JPM", "BAC"), ("JPM", "C"), ("BAC", "C"), ("WFC", "JPM"), ("GS", "MS"), ("XOM", "CVX"), ("COP", "XOM"), ("SLB", "HAL"),
         ("OXY", "COP"), ("KO", "PEP"), ("PG", "CL"), ("MO", "PM"), ("HD", "LOW"), ("WMT", "TGT"), ("COST", "WMT"), ("TXN", "ADI"),
         ("QCOM", "TXN"), ("INTC", "TXN"), ("V", "MA"), ("UPS", "FDX"), ("UNP", "CSX"), ("NSC", "CSX"), ("PFE", "MRK"), ("JNJ", "PFE"),
         ("ABT", "JNJ"), ("T", "VZ"), ("MCD", "YUM"), ("EWA", "EWC"), ("EWG", "EWU"), ("XLE", "XOP"), ("GLD", "GDX"), ("SPY", "IVV")]
SYMBOLS = sorted({s for p in PAIRS for s in p} | {"^VIX"})
