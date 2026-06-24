#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SMALL-CAP AAR / CAAR PLOT  (response to supervisor comment on Section 6.1.3)
================================================================================

The supervisor noted that a near-zero average CAR over a *balanced* [-1,+1]
window can hide large offsetting daily spikes, and asked for a plot of the
average abnormal return (AAR) against event time (e.g. days -10..+10 or -5..+5)
to show (a) whether positive and negative daily reactions cancel out and
(b) whether the news was anticipated and priced in *before* the event window.

This script reuses the EXACT small-cap universe and methodology from
`robustness_smallcap_16country.py` (same 114 firms, same local indices, same
[-250,-11] estimation window, same Dimson (1979) lead/lag betas) and simply
widens the event window so it can report a daily abnormal-return path:

    * AAR_t   = average market-model abnormal return across firms on event day t
    * CAAR_t  = running cumulative sum of AAR from the start of the plot window

It then:
    1. saves  smallcap_aar_caar.csv   (AAR, CAAR and firm count per event day)
    2. saves  smallcap_aar_plot.png   (bars = AAR, line = CAAR, shaded [-1,+1])
    3. prints the key numbers you need for the Section 6.1.3 paragraph
       (pre-event CAAR, in-window CAAR, largest positive / negative AAR days).

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
    pip install yfinance pandas numpy statsmodels matplotlib
    (the same environment you already use to run robustness_smallcap_16country.py)

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------
    # from the folder that contains robustness_smallcap_16country.py
    python smallcap_aar_plot.py

Change PLOT_LO / PLOT_HI below to switch between a [-10,+10] and a [-5,+5] window.
Everything else (tickers, indices, betas) comes straight from your existing
script, so the figure is fully consistent with Table 8 / Table 9.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")          # write a file without needing a display
import matplotlib.pyplot as plt

# --- make sure we can import your existing script from this folder ----------- #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robustness_smallcap_16country as base   # reuse universe + helpers + params

# ------------------------------------------------------------------ #
# PARAMETERS  (inherit the thesis settings; only the window widens)   #
# ------------------------------------------------------------------ #
EVENT_DATE  = base.EVENT_DATE          # 2025-04-02
EST_START   = base.EST_START           # -250
EST_END     = base.EST_END             # -11
USE_DIMSON  = base.USE_DIMSON          # True -> Dimson lead/lag betas
UNIVERSE    = base.UNIVERSE            # [(ticker, country, index_ticker), ...]

PLOT_LO, PLOT_HI = -10, +10            # plot window; set to -5, +5 for the shorter view
EVT_LO,  EVT_HI  = -1, +1             # the headline event window (shaded in the plot)

OUT_CSV = "smallcap_aar_caar.csv"
OUT_PNG = "smallcap_aar_plot.png"


# ------------------------------------------------------------------ #
# 1. DAILY ABNORMAL-RETURN PATH FOR ONE FIRM                          #
# ------------------------------------------------------------------ #
def firm_ar_path(firm_ret, mkt_ret, lo, hi):
    """
    Estimate the market model (Dimson optional) on [-250,-11] exactly as in the
    thesis, then return abnormal returns for every event day in [lo, hi] as a
    Series indexed by integer event-day offset. Returns None if coverage is
    insufficient (same coverage rule as the main script).
    """
    df = pd.concat([firm_ret, mkt_ret], axis=1, keys=["r", "rm"]).dropna()
    if df.empty:
        return None
    dates = df.index
    pos = dates.searchsorted(EVENT_DATE, side="right") - 1     # last trading day <= event
    if pos < 0:
        return None
    est_lo, est_hi = pos + EST_START, pos + EST_END
    evt_lo, evt_hi = pos + lo, pos + hi
    need_lo = est_lo - (1 if USE_DIMSON else 0)
    if need_lo < 0 or evt_hi >= len(df) or evt_lo < 0:
        return None

    est = df.iloc[est_lo:est_hi + 1].copy()
    if USE_DIMSON:
        est["rm_lag"]  = df["rm"].shift(1).reindex(est.index)
        est["rm_lead"] = df["rm"].shift(-1).reindex(est.index)
        est = est.dropna()
        X = sm.add_constant(est[["rm_lag", "rm", "rm_lead"]])
        m = sm.OLS(est["r"], X).fit()
        alpha = m.params["const"]
        beta  = m.params["rm_lag"] + m.params["rm"] + m.params["rm_lead"]
    else:
        X = sm.add_constant(est["rm"])
        m = sm.OLS(est["r"], X).fit()
        alpha, beta = m.params["const"], m.params["rm"]

    ew = df.iloc[evt_lo:evt_hi + 1]
    ar = (ew["r"] - (alpha + beta * ew["rm"])) * 100.0          # in percentage points
    ar.index = range(lo, hi + 1)                                # event-day offsets
    return ar


# ------------------------------------------------------------------ #
# 2. BUILD THE AAR / CAAR PATH ACROSS ALL FIRMS                       #
# ------------------------------------------------------------------ #
def build_aar():
    index_cache = {}
    rows, n_ok = [], 0
    for ticker, country, idx_ticker in UNIVERSE:
        px = base.fetch_prices(ticker)
        if px is None:
            print(f"  [skip] {ticker:<12} ({country}) - no price data")
            continue
        if idx_ticker not in index_cache:
            ipx = base.fetch_prices(idx_ticker)
            index_cache[idx_ticker] = base.log_returns(ipx) if ipx is not None else None
        mkt = index_cache[idx_ticker]
        if mkt is None:
            print(f"  [skip] {ticker:<12} ({country}) - no index data ({idx_ticker})")
            continue
        ar = firm_ar_path(base.log_returns(px), mkt, PLOT_LO, PLOT_HI)
        if ar is None:
            print(f"  [skip] {ticker:<12} ({country}) - insufficient window coverage")
            continue
        rows.append(ar.rename(ticker))
        n_ok += 1

    if not rows:
        raise SystemExit("No firms produced an abnormal-return path; check tickers / data.")

    mat = pd.concat(rows, axis=1)                     # rows = event day, cols = firm
    aar  = mat.mean(axis=1)                            # average abnormal return per day
    caar = aar.cumsum()                               # cumulative average abnormal return
    n_by_day = mat.notna().sum(axis=1)
    print(f"\nFirms included: {n_ok} of {len(UNIVERSE)}")
    out = pd.DataFrame({"event_day": aar.index, "AAR_pct": aar.values,
                        "CAAR_pct": caar.values, "N_firms": n_by_day.values})
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")
    return aar, caar


# ------------------------------------------------------------------ #
# 3. PLOT                                                             #
# ------------------------------------------------------------------ #
def plot(aar, caar):
    days = list(aar.index)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # shade the headline [-1,+1] event window
    ax1.axvspan(EVT_LO - 0.5, EVT_HI + 0.5, color="0.85", zorder=0,
                label="Event window [-1, +1]")
    ax1.axhline(0, color="0.4", lw=0.8)
    ax1.axvline(0, color="0.4", lw=0.8, ls=":")

    bars = ax1.bar(days, aar.values, width=0.7, color="#4C72B0",
                   alpha=0.85, label="AAR (left axis)")
    ax1.set_xlabel("Event day relative to 2 April 2025 (t = 0)")
    ax1.set_ylabel("Average abnormal return, AAR (%)")
    ax1.set_xticks(days)

    ax2 = ax1.twinx()
    ax2.plot(days, caar.values, color="#C44E52", marker="o", lw=2,
             label="CAAR (right axis)")
    ax2.set_ylabel("Cumulative average abnormal return, CAAR (%)")

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9, fontsize=9)

    plt.title("Small-cap sample: daily average abnormal returns around the\n"
              "2 April 2025 reciprocal-tariff announcement", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    print(f"Saved -> {OUT_PNG}")


# ------------------------------------------------------------------ #
# 4. NUMBERS FOR THE WRITE-UP                                         #
# ------------------------------------------------------------------ #
def summarise(aar, caar):
    pre = aar.loc[PLOT_LO:EVT_LO - 1].sum()        # cumulative AAR before the window
    win = aar.loc[EVT_LO:EVT_HI].sum()             # cumulative AAR inside [-1,+1]
    post = aar.loc[EVT_HI + 1:PLOT_HI].sum()       # cumulative AAR after the window
    pos_day = aar.idxmax(); neg_day = aar.idxmin()
    print("\n" + "=" * 64)
    print("NUMBERS FOR THE SECTION 6.1.3 PARAGRAPH")
    print("=" * 64)
    print(f"  Largest positive AAR : {aar.max():+.3f}%  on event day {pos_day:+d}")
    print(f"  Largest negative AAR : {aar.min():+.3f}%  on event day {neg_day:+d}")
    print(f"  Cumulative AAR before window  [{PLOT_LO},{EVT_LO-1}] : {pre:+.3f}%")
    print(f"  Cumulative AAR inside window  [{EVT_LO},{EVT_HI}]   : {win:+.3f}%")
    print(f"  Cumulative AAR after window   [{EVT_HI+1},{PLOT_HI}] : {post:+.3f}%")
    print(f"  Total CAAR over [{PLOT_LO},{PLOT_HI}]              : {caar.iloc[-1]:+.3f}%")
    print("=" * 64)


if __name__ == "__main__":
    print("Building small-cap AAR/CAAR path  |  Dimson:",
          "ON" if USE_DIMSON else "OFF",
          f"|  window [{PLOT_LO},{PLOT_HI}]")
    aar, caar = build_aar()
    plot(aar, caar)
    summarise(aar, caar)
    print("\nDone.")
