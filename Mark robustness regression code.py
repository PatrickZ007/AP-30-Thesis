#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-run the small-cap robustness with the CORRECT control set (3 controls only):
    ln(Total assets), Leverage, Book-to-market
ROA, Cash/Assets and Volatility are DROPPED (per Patrick: Zc deleted, Xi 5->3).

Reads the existing firm-level panel (no Yahoo download), so it reproduces the
exact same CARs/controls used before -- only the regression control set changes.

    pip install pandas numpy statsmodels scipy
    python robustness_3controls_rerun.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

INFILE = "smallcap_firm_level.csv"   # same folder as this script

# Unweighted 16-country mean tariff = the demeaning constant (= 23.125 -> "23.1")
COUNTRY_TARIFF = {
    "Thailand": 36, "China": 34, "Indonesia": 32, "Taiwan": 32,
    "Switzerland": 31, "South Africa": 30, "India": 26, "South Korea": 25,
    "Japan": 24, "France": 20, "Germany": 20, "Netherlands": 20,
    "Australia": 10, "Brazil": 10, "Singapore": 10, "United Kingdom": 10,
}
MEAN_TARIFF = float(np.mean(list(COUNTRY_TARIFF.values())))

CONTROLS = ["ln_assets", "leverage", "book_to_market"]   # <-- the 3 correct controls

panel = pd.read_csv(INFILE)
print(f"Loaded {len(panel)} firms.  Demeaning constant (16-country mean tariff) = {MEAN_TARIFF:.1f}%")
print(f"Controls used: {CONTROLS}\n")


# ------------------------------------------------------------------ #
# DESCRIPTIVES  (Panel A by country; Panel B summary stats)          #
# ------------------------------------------------------------------ #
def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

car = panel["CAR"].dropna()
n = len(car)
tstat = car.mean() / (car.std(ddof=1) / np.sqrt(n))
print("========== AGGREGATE MARKET REACTION ==========")
print(f"  N firms            : {n}")
print(f"  Mean CAR[-1,+1]    : {car.mean():+.4f} %")
print(f"  Median CAR[-1,+1]  : {car.median():+.4f} %")
print(f"  t-statistic (H0=0) : {tstat:+.3f}")
print(f"  Share negative     : {100*(car<0).mean():.1f} %")

pa = (panel.groupby("country")
      .agg(N=("CAR", "size"), Tariff=("tariff", "first"), FDc=("FDc", "first"),
           Mean_CAR=("CAR", "mean"), Median_CAR=("CAR", "median"))
      .reset_index())

vars_ = ["CAR", "tariff", "FDc", "ln_assets", "leverage", "book_to_market"]  # 3 controls only
pb = []
for v in vars_:
    s = panel[v].dropna()
    jb, jbp = stats.jarque_bera(s) if len(s) > 7 else (np.nan, np.nan)
    pb.append({"Variable": v, "N": len(s), "Mean": s.mean(), "Median": s.median(),
               "Std": s.std(ddof=1), "Min": s.min(), "Max": s.max(),
               "Skew": stats.skew(s), "ExKurt": stats.kurtosis(s), "JB": jb, "JB_p": jbp})
pb = pd.DataFrame(pb)

print("\n========== PANEL A: BY COUNTRY ==========")
print(pa.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n========== PANEL B: SUMMARY STATISTICS ==========")
print(pb.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ------------------------------------------------------------------ #
# CROSS-SECTIONAL REGRESSIONS (Table 9, 3-control version)           #
# ------------------------------------------------------------------ #
df = panel.dropna(subset=["CAR"]).copy()
df["tariff_shock"] = df["tariff"] - MEAN_TARIFF
df["fd_x_shock"] = df["FDc"] * df["tariff_shock"]

specs = {
    "(1)": ["tariff_shock"],
    "(2)": ["tariff_shock"] + CONTROLS,
    "(3)": ["tariff_shock", "FDc", "fd_x_shock"],
    "(4)": ["tariff_shock", "FDc", "fd_x_shock"] + CONTROLS,
}

results = {}
for name, regs in specs.items():
    d = df.dropna(subset=regs + ["CAR"]).copy()
    X = sm.add_constant(d[regs])
    m = sm.OLS(d["CAR"], X).fit(cov_type="cluster", cov_kwds={"groups": d["country"]})
    results[name] = (m, d.shape[0])

order = ["tariff_shock", "FDc", "fd_x_shock"] + CONTROLS + ["const"]
label = {"tariff_shock": "Tariff shock", "FDc": "FDc", "fd_x_shock": "FDc x Tariff shock",
         "ln_assets": "ln(Total assets)", "leverage": "Leverage",
         "book_to_market": "Book-to-market", "const": "Constant"}

print("\n========== TABLE 9 (CORRECTED): CROSS-SECTIONAL REGRESSIONS, CAR[-1,+1] % ==========")
for var in order:
    coefs, ses = [], []
    for name in ["(1)", "(2)", "(3)", "(4)"]:
        m, _ = results[name]
        if var in m.params.index:
            coefs.append(f"{m.params[var]:.4f}{stars(m.pvalues[var])}")
            ses.append(f"({m.bse[var]:.4f})")
        else:
            coefs.append(""); ses.append("")
    print(f"{label[var]:>20s}  " + "  ".join(f"{c:>12s}" for c in coefs))
    print(f"{'':>20s}  " + "  ".join(f"{c:>12s}" for c in ses))
print("-" * 78)
print(f"{'Observations':>20s}  " + "  ".join(f"{results[n][1]:>12d}" for n in ['(1)','(2)','(3)','(4)']))
print(f"{'R-squared':>20s}  " + "  ".join(f"{results[n][0].rsquared:>12.4f}" for n in ['(1)','(2)','(3)','(4)']))
print(f"{'Adj. R-squared':>20s}  " + "  ".join(f"{results[n][0].rsquared_adj:>12.4f}" for n in ['(1)','(2)','(3)','(4)']))

# marginal-effect crossing point from the full model (4): dCAR/dshock = b1 + b3*FDc = 0
m4 = results["(4)"][0]
b1, b3 = m4.params["tariff_shock"], m4.params["fd_x_shock"]
print(f"\nMarginal effect (model 4): dCAR/dShock = {b1:.4f} + {b3:.4f} * FDc"
      f"   -> crosses zero at FDc = {(-b1/b3):.3f}")


# ------------------------------------------------------------------ #
# WILD CLUSTER BOOTSTRAP (Cameron, Gelbach & Miller 2008)            #
#   null-imposed, Rademacher, one-sided; full 3-control model        #
# ------------------------------------------------------------------ #
def _design(d, regs):
    return np.column_stack([np.ones(len(d))] + [d[r].values.astype(float) for r in regs])

def _cluster_t(Xf, y, codes, n_groups, XtX_inv, P, k):
    N, K = Xf.shape
    beta = P @ y
    u = y - Xf @ beta
    meat = np.zeros((K, K))
    for gi in range(n_groups):
        idx = codes == gi
        s = Xf[idx].T @ u[idx]
        meat += np.outer(s, s)
    c = (n_groups / (n_groups - 1.0)) * ((N - 1.0) / (N - K))
    V = c * (XtX_inv @ meat @ XtX_inv)
    return beta[k] / np.sqrt(V[k, k]), beta[k], np.sqrt(V[k, k])

def wild_cluster_bootstrap(d, regs, coef, tail, B=9999, seed=12345):
    d = d.dropna(subset=regs + ["CAR"]).copy()
    y = d["CAR"].values.astype(float)
    Xf = _design(d, regs)
    names = ["const"] + regs
    k = names.index(coef)
    cats = d["country"].astype("category")
    codes = cats.cat.codes.values
    n_groups = len(cats.cat.categories)
    XtX_inv = np.linalg.inv(Xf.T @ Xf)
    P = XtX_inv @ Xf.T
    t_orig, b_orig, se_orig = _cluster_t(Xf, y, codes, n_groups, XtX_inv, P, k)
    r_regs = [r for r in regs if r != coef]
    Xr = _design(d, r_regs)
    beta_r = np.linalg.inv(Xr.T @ Xr) @ Xr.T @ y
    yhat_r, u_r = Xr @ beta_r, y - Xr @ beta_r
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(B):
        w = (rng.integers(0, 2, size=n_groups) * 2 - 1)[codes].astype(float)
        y_star = yhat_r + w * u_r
        t_b, _, _ = _cluster_t(Xf, y_star, codes, n_groups, XtX_inv, P, k)
        if (t_b >= t_orig) if tail == "right" else (t_b <= t_orig):
            count += 1
    return t_orig, b_orig, se_orig, count / B

full = ["tariff_shock", "FDc", "fd_x_shock"] + CONTROLS
print("\n========== WILD CLUSTER BOOTSTRAP (full 3-control model, 9999 reps) ==========")
t3, b3b, se3, p3 = wild_cluster_bootstrap(df, full, "fd_x_shock", tail="right")
print(f"  H2  FDc x Tariff shock : coef={b3b:+.4f}  t={t3:+.3f}  one-sided bootstrap p = {p3:.4f}")
t1, b1b, se1, p1 = wild_cluster_bootstrap(df, full, "tariff_shock", tail="left")
print(f"  H1  Tariff shock       : coef={b1b:+.4f}  t={t1:+.3f}  one-sided bootstrap p = {p1:.4f}")

print("\nDone. Copy this entire console output back to me and I'll update the tables and the write-up.")
