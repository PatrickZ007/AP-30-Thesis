#!/usr/bin/env python3
"""
=================================================================
Regression Analysis: CAR[0, +5] — Trump Tariffs & Financial Resilience
=================================================================
Merges CAR[0,+5] from car_0_5_results.xlsx with firm-level data
from dependent_variables.xlsx, then runs H1/H2 regressions.

Usage:
    pip install pandas numpy statsmodels scipy matplotlib seaborn openpyxl
    python regression_car05.py
=================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from scipy.stats.mstats import winsorize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# ============================================================
# CONFIGURATION
# ============================================================
CAR_FILE = "car_0_5_results.xlsx"        # CAR[0,+5] from event study
DATA_FILE = "dependent_variables.xlsx"    # firm financials + tariff + FDc
BASELINE_TARIFF = 0.10
WINSOR_LIMITS = [0.01, 0.01]
CONTROLS = ["ln_Total_Assets", "BM_Ratio", "Leverage"]
OSTER_RMAX_FACTOR = 1.3


# ============================================================
# UTILITY
# ============================================================
def stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""

def one_sided_p(t_stat, df, tail="left"):
    if tail == "left":
        return stats.t.cdf(t_stat, df)
    else:
        return 1 - stats.t.cdf(t_stat, df)

def section(title):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


# ============================================================
# 1. LOAD & MERGE
# ============================================================
def load_and_merge():
    # CAR[0,+5]
    car = pd.read_excel(CAR_FILE)
    car = car[["Ticker", "CAR"]].rename(columns={"CAR": "CAR_05"})
    car = car.dropna(subset=["CAR_05"])

    # firm-level data (tariff, FDc, financials)
    fin = pd.read_excel(DATA_FILE, sheet_name="Sheet1")
    fin = fin.dropna(axis=1, how="all")

    # merge on Ticker
    df = fin.merge(car, on="Ticker", how="inner")

    # replace CAR with CAR[0,+5]
    df["CAR"] = df["CAR_05"]
    df = df.drop(columns=["CAR_05"])

    # create variables
    df["ExcessTariff"] = df["tariff"] - BASELINE_TARIFF
    return df


# ============================================================
# 2. NORMALITY TESTS
# ============================================================
def normality_tests(df, cols, label=""):
    print(f"\n  Normality tests — {label}")
    print(f"  {'Variable':<20s} {'Shapiro W':>10s} {'SW p':>10s} "
          f"{'JB stat':>10s} {'JB p':>10s} {'Normal?':>8s}")
    print("  " + "-" * 68)
    for c in cols:
        x = df[c].dropna()
        sw_stat, sw_p = stats.shapiro(x)
        jb_stat, jb_p = stats.jarque_bera(x)
        normal = "Yes" if (sw_p > 0.05 and jb_p > 0.05) else "No"
        print(f"  {c:<20s} {sw_stat:10.4f} {sw_p:10.4f} "
              f"{jb_stat:10.4f} {jb_p:10.4f} {normal:>8s}")


def plot_histograms(df, cols, filename, suptitle=""):
    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, c in enumerate(cols):
        ax = axes_flat[i]
        ax.hist(df[c].dropna(), bins=20, density=True,
                alpha=0.6, color="steelblue", edgecolor="white")
        df[c].dropna().plot.kde(ax=ax, color="darkred", linewidth=1.5)
        ax.set_title(c, fontsize=10)
        ax.set_ylabel("")
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ============================================================
# 3. WINSORIZE + LN
# ============================================================
def treat_data(df):
    df = df.copy()
    df["ln_Total_Assets"] = np.log(df["Total_Assets_USD"])
    for c in CONTROLS:
        arr = winsorize(df[c].values, limits=WINSOR_LIMITS)
        df[c] = np.array(arr)
    df["FD_x_ExTariff"] = df["FDc"] * df["ExcessTariff"]
    return df


# ============================================================
# 4. CORRELATION + HEATMAP
# ============================================================
def correlation_analysis(df):
    cols = ["CAR", "ExcessTariff", "FDc", "FD_x_ExTariff"] + CONTROLS
    corr = df[cols].corr()
    print("\n  Pearson correlation matrix")
    print(corr.round(3).to_string())

    print("\n  Pairs with |r| > 0.7:")
    flagged = False
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                print(f"    {cols[i]} × {cols[j]} : r = {r:.3f}")
                flagged = True
    if not flagged:
        print("    (none)")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation — CAR[0,+5]", fontsize=12)
    fig.tight_layout()
    fig.savefig("05_corr_heatmap_car05.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 05_corr_heatmap_car05.png")


# ============================================================
# 5. VIF
# ============================================================
def calc_vif(X_df, model_name):
    print(f"\n  VIF — {model_name}")
    print(f"  {'Variable':<20s} {'VIF':>8s}")
    print("  " + "-" * 30)
    if len(X_df.columns) < 2:
        print(f"  {X_df.columns[0]:<20s} {'1.00':>8s}  (single regressor)")
        return
    X_c = sm.add_constant(X_df)
    for i, col in enumerate(X_c.columns):
        if col == "const":
            continue
        v = variance_inflation_factor(X_c.values, i)
        flag = " ⚠" if v > 10 else ""
        print(f"  {col:<20s} {v:8.2f}{flag}")


# ============================================================
# 6. OLS REGRESSION
# ============================================================
def build_models(df):
    y = df["CAR"]
    groups = df["Country"]
    specs = {
        "H1a": ["ExcessTariff"],
        "H1b": ["ExcessTariff"] + CONTROLS,
        "H2a": ["ExcessTariff", "FDc", "FD_x_ExTariff"],
        "H2b": ["ExcessTariff", "FDc", "FD_x_ExTariff"] + CONTROLS,
    }
    results = {}
    for name, xvars in specs.items():
        X = sm.add_constant(df[xvars])
        model = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
        results[name] = model
    return specs, results


def print_regression_table(specs, results, n_clusters):
    models = list(specs.keys())
    all_vars = []
    for m in models:
        for v in ["const"] + specs[m]:
            if v not in all_vars:
                all_vars.append(v)

    labels = {
        "const": "Constant", "ExcessTariff": "Excess Tariff",
        "FDc": "FD", "FD_x_ExTariff": "FD × Excess Tariff",
        "ln_Total_Assets": "ln(Total Assets)",
        "BM_Ratio": "Book-to-Market", "Leverage": "Leverage",
    }

    cw, lw = 16, 22
    sep = "-" * (lw + cw * len(models))
    print(f"\n{sep}")
    print(f"{'Dep: CAR[0,+5]':>{lw}}" +
          "".join(f"{'(' + str(i+1) + ')':>{cw}}" for i in range(len(models))))
    print(f"{'':>{lw}}" + "".join(f"{m:>{cw}}" for m in models))
    print(sep)

    for v in all_vars:
        cl = f"{labels.get(v,v):>{lw}}"
        sl = f"{'':>{lw}}"
        for m in models:
            res = results[m]
            if v in res.params.index:
                b, p, se = res.params[v], res.pvalues[v], res.bse[v]
                cl += f"{b:>{cw-3}.4f}{stars(p):>3s}"
                sl += f"{'(' + f'{se:.4f}' + ')':>{cw}}"
            else:
                cl += f"{'':>{cw}}"
                sl += f"{'':>{cw}}"
        print(cl)
        print(sl)

    print(sep)
    for sn, gt in [
        ("N",        lambda r: f"{int(r.nobs)}"),
        ("R²",       lambda r: f"{r.rsquared:.4f}"),
        ("Adj. R²",  lambda r: f"{r.rsquared_adj:.4f}"),
        ("F-stat",   lambda r: f"{r.fvalue:.2f}"),
        ("Clusters", lambda r: f"{n_clusters}"),
    ]:
        row = f"{sn:>{lw}}"
        for m in models:
            row += f"{gt(results[m]):>{cw}}"
        print(row)
    print(sep)
    print("  Clustered SE (by country) in parentheses.")
    print("  *** p<0.01, ** p<0.05, * p<0.10 (two-sided)")


# ============================================================
# 7. ONE-SIDED TESTS
# ============================================================
def hypothesis_tests(results):
    print("\n  H1: β₁(ExcessTariff) — left-tailed (β₁ < 0)")
    print(f"  {'Model':<8s} {'β₁':>10s} {'t-stat':>10s} "
          f"{'p(two)':>10s} {'p(one-L)':>10s} {'Sig':>6s}")
    print("  " + "-" * 56)
    for m in ["H1a", "H1b", "H2a", "H2b"]:
        res = results[m]
        b = res.params["ExcessTariff"]
        t = res.tvalues["ExcessTariff"]
        p2 = res.pvalues["ExcessTariff"]
        p1 = one_sided_p(t, res.df_resid, tail="left")
        print(f"  {m:<8s} {b:10.4f} {t:10.4f} {p2:10.4f} {p1:10.4f} {stars(p1):>6s}")

    print("\n  H2: β₃(FD × ExcessTariff) — right-tailed (β₃ > 0)")
    print(f"  {'Model':<8s} {'β₃':>10s} {'t-stat':>10s} "
          f"{'p(two)':>10s} {'p(one-R)':>10s} {'Sig':>6s}")
    print("  " + "-" * 56)
    for m in ["H2a", "H2b"]:
        res = results[m]
        b = res.params["FD_x_ExTariff"]
        t = res.tvalues["FD_x_ExTariff"]
        p2 = res.pvalues["FD_x_ExTariff"]
        p1 = one_sided_p(t, res.df_resid, tail="right")
        print(f"  {m:<8s} {b:10.4f} {t:10.4f} {p2:10.4f} {p1:10.4f} {stars(p1):>6s}")


# ============================================================
# 8. HETEROSKEDASTICITY
# ============================================================
def het_tests(specs, results):
    print(f"\n  {'Model':<8s} {'BP stat':>10s} {'BP p':>10s} "
          f"{'White stat':>12s} {'White p':>10s} {'DW':>8s}")
    print("  " + "-" * 62)
    for m in specs:
        res = results[m]
        bp_stat, bp_p, _, _ = het_breuschpagan(res.resid, res.model.exog)
        try:
            w_stat, w_p, _, _ = het_white(res.resid, res.model.exog)
        except:
            w_stat, w_p = np.nan, np.nan
        dw = durbin_watson(res.resid)
        print(f"  {m:<8s} {bp_stat:10.4f} {bp_p:10.4f} "
              f"{w_stat:12.4f} {w_p:10.4f} {dw:8.4f}")
    print("  BP / White H₀: homoskedasticity. Reject if p < 0.05.")


# ============================================================
# 9. RESIDUAL DIAGNOSTICS
# ============================================================
def residual_diagnostics(specs, results):
    print(f"\n  Residual normality")
    print(f"  {'Model':<8s} {'Shapiro W':>10s} {'p-value':>10s} "
          f"{'JB stat':>10s} {'JB p':>10s} {'Normal?':>8s}")
    print("  " + "-" * 58)

    models_list = list(specs.keys())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, m in enumerate(models_list):
        resid = results[m].resid
        sw_stat, sw_p = stats.shapiro(resid)
        jb_stat, jb_p = stats.jarque_bera(resid)
        normal = "Yes" if (sw_p > 0.05 and jb_p > 0.05) else "No"
        print(f"  {m:<8s} {sw_stat:10.4f} {sw_p:10.4f} "
              f"{jb_stat:10.4f} {jb_p:10.4f} {normal:>8s}")
        stats.probplot(resid, dist="norm", plot=axes[i])
        axes[i].set_title(f"{m} residuals", fontsize=10)
    fig.suptitle("Q-Q Plots — CAR[0,+5] Residuals", fontsize=12)
    fig.tight_layout()
    fig.savefig("06_qq_car05.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 06_qq_car05.png")


# ============================================================
# 10. RAMSEY RESET
# ============================================================
def ramsey_reset_test(specs, results):
    print(f"\n  {'Model':<8s} {'F-stat':>10s} {'p-value':>10s} "
          f"{'Conclusion':>20s}")
    print("  " + "-" * 52)
    for m in specs:
        try:
            reset = linear_reset(results[m], power=3, use_f=True)
            concl = "Misspec. detected" if reset.pvalue < 0.05 else "No misspec."
            print(f"  {m:<8s} {reset.fvalue:10.4f} {reset.pvalue:10.4f} "
                  f"{concl:>20s}")
        except Exception as e:
            print(f"  {m:<8s} {'N/A':>10s} {'N/A':>10s} {'Error':>20s}")


# ============================================================
# 11. COEFFICIENT STABILITY + OSTER δ
# ============================================================
def coefficient_stability(results):
    print("\n  A. Coefficient stability across models")
    print(f"  {'Model':<8s} {'β₁(ExTariff)':>14s} {'SE':>10s} "
          f"{'R²':>8s} {'Adj.R²':>8s}")
    print("  " + "-" * 52)
    for m in ["H1a", "H1b", "H2a", "H2b"]:
        res = results[m]
        b, se = res.params["ExcessTariff"], res.bse["ExcessTariff"]
        print(f"  {m:<8s} {b:14.6f} {se:10.6f} "
              f"{res.rsquared:8.4f} {res.rsquared_adj:8.4f}")

    print("\n  B. Oster (2019) δ")
    print("  |δ| > 1 → robust to omitted variable bias\n")

    def oster_delta(b_s, r2_s, b_l, r2_l, r_max):
        d = (b_s - b_l) * (r_max - r2_l)
        return (b_l * (r2_l - r2_s)) / d if abs(d) > 1e-12 else np.nan

    def oster_bstar(b_s, r2_s, b_l, r2_l, r_max):
        d = r2_l - r2_s
        return b_l - (b_s - b_l) * (r_max - r2_l) / d if abs(d) > 1e-12 else np.nan

    pairs = [
        ("H1: β₁", "H1a", "H1b", "ExcessTariff"),
        ("H2: β₁", "H2a", "H2b", "ExcessTariff"),
        ("H2: β₃", "H2a", "H2b", "FD_x_ExTariff"),
    ]

    print(f"  {'Coef':<12s} {'β_short':>10s} {'β_long':>10s} "
          f"{'R²_s':>8s} {'R²_l':>8s} {'R_max':>8s} "
          f"{'δ*':>8s} {'β*(δ=1)':>10s} {'Robust?':>8s}")
    print("  " + "-" * 86)

    for label, short, long, var in pairs:
        rs, rl = results[short], results[long]
        b_s, b_l = rs.params[var], rl.params[var]
        r2_s, r2_l = rs.rsquared, rl.rsquared
        r_max = min(1.0, OSTER_RMAX_FACTOR * r2_l)
        delta = oster_delta(b_s, r2_s, b_l, r2_l, r_max)
        bstar = oster_bstar(b_s, r2_s, b_l, r2_l, r_max)
        robust = "Yes" if (pd.notna(delta) and abs(delta) > 1) else "No"
        d_s = f"{delta:8.3f}" if pd.notna(delta) else f"{'N/A':>8s}"
        b_s_str = f"{bstar:10.6f}" if pd.notna(bstar) else f"{'N/A':>10s}"
        print(f"  {label:<12s} {b_s:10.6f} {b_l:10.6f} "
              f"{r2_s:8.4f} {r2_l:8.4f} {r_max:8.4f} "
              f"{d_s} {b_s_str} {robust:>8s}")

    print(f"\n  R_max = min(1, {OSTER_RMAX_FACTOR} × R²_long)")


# ============================================================
# MAIN
# ============================================================
def main():
    # ── 1. load & merge ──
    section("1. DATA LOADING — CAR[0,+5]")
    df = load_and_merge()
    print(f"  Merged observations : {len(df)}")
    print(f"  Countries           : {df['Country'].nunique()}")
    print(f"  ExcessTariff range  : [{df['ExcessTariff'].min():.2f}, "
          f"{df['ExcessTariff'].max():.2f}]")
    print(f"  CAR[0,+5] mean      : {df['CAR'].mean():.6f}")

    # ── 2. normality BEFORE ──
    section("2. NORMALITY — RAW CONTROLS")
    raw_cols = ["Total_Assets_USD", "BM_Ratio", "Leverage"]
    normality_tests(df, raw_cols, label="before treatment")
    plot_histograms(df, raw_cols, "05_hist_raw_car05.png",
                    suptitle="Controls — Before Treatment (CAR[0,+5] sample)")

    # ── 3. treat ──
    section("3. LN + WINSORIZE (1%/99%)")
    df = treat_data(df)
    print(f"  ln(Total_Assets_USD) → ln_Total_Assets")
    print(f"  Winsorized: {', '.join(CONTROLS)}")

    # ── 4. normality AFTER ──
    section("4. NORMALITY — TREATED CONTROLS")
    normality_tests(df, CONTROLS, label="after treatment")
    plot_histograms(df, CONTROLS, "05_hist_treated_car05.png",
                    suptitle="Controls — After Treatment (CAR[0,+5] sample)")

    # ── 5. descriptive ──
    section("5. DESCRIPTIVE STATISTICS")
    print(df[["CAR", "ExcessTariff", "FDc"] + CONTROLS]
          .describe().round(4).to_string())

    # ── 6. correlation ──
    section("6. CORRELATION MATRIX")
    correlation_analysis(df)

    # ── 7. VIF ──
    section("7. VIF")
    specs_vif = {
        "H1a": ["ExcessTariff"],
        "H1b": ["ExcessTariff"] + CONTROLS,
        "H2a": ["ExcessTariff", "FDc", "FD_x_ExTariff"],
        "H2b": ["ExcessTariff", "FDc", "FD_x_ExTariff"] + CONTROLS,
    }
    for m, xv in specs_vif.items():
        calc_vif(df[xv], m)

    # ── 8. regression ──
    section("8. OLS — COUNTRY-CLUSTERED SE")
    specs, results = build_models(df)
    print_regression_table(specs, results, df["Country"].nunique())

    # ── 9. one-sided ──
    section("9. ONE-SIDED HYPOTHESIS TESTS")
    hypothesis_tests(results)

    # ── 10. heteroskedasticity ──
    section("10. HETEROSKEDASTICITY")
    het_tests(specs, results)

    # ── 11. residuals ──
    section("11. RESIDUAL DIAGNOSTICS")
    residual_diagnostics(specs, results)

    # ── 12. RESET ──
    section("12. RAMSEY RESET")
    ramsey_reset_test(specs, results)

    # ── 13. stability + Oster ──
    section("13. COEFFICIENT STABILITY & OSTER δ")
    coefficient_stability(results)

    section("COMPLETE")
    print("  PNG outputs:")
    print("    05_hist_raw_car05.png")
    print("    05_hist_treated_car05.png")
    print("    05_corr_heatmap_car05.png")
    print("    06_qq_car05.png")
    print()


if __name__ == "__main__":
    main()
