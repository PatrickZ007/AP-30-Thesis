#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ROBUSTNESS ANALYSIS: SMALL-CAP REPLICATION OF THE APRIL 2, 2025 TARIFF EVENT STUDY
                      ---  UPDATED FOR THE 16-COUNTRY / 114-FIRM MAIN SAMPLE  ---
================================================================================

This script reproduces the EXACT methodology of the main thesis on a sample of
SMALL-CAP firms instead of the large-cap firms used in the headline analysis,
holding everything else constant:

  * same 16 countries and the same number of firms per country (114 in total)
  * same local market indices
  * same estimation window [-250, -11] and event window [-1, +1]
  * same market-model abnormal returns and CAR[-1,+1]
  * same firm-level controls, tariff shock and financial-development measure
  * same four nested cross-sectional OLS regressions with country-clustered SEs

Addition for the small-cap setting: betas can be estimated with the Dimson (1979)
one-lead / one-lag correction to mitigate the downward bias caused by the thin,
non-synchronous trading typical of small firms. Toggle with USE_DIMSON below.

--------------------------------------------------------------------------------
WHAT CHANGED VS. THE PREVIOUS (12-COUNTRY) VERSION
--------------------------------------------------------------------------------
  * Added FOUR countries that are now in the headline sample: Thailand,
    Indonesia, South Africa and the Netherlands (tariff, FDc, local index below).
  * The demeaned tariff shock is now the country tariff minus the unweighted
    SIXTEEN-country mean (= 23.1%, computed automatically), not 21.0%.
  * The original twelve countries' candidate tickers are UNCHANGED (you already
    validated those).  The four new countries' tickers are UNVERIFIED PLACEHOLDERS
    -- confirm each one via the coverage report before reporting any number.

  >>> IMPORTANT: this rebuild keeps your original twelve-country composition and
  >>> ADDS the four new countries.  If your final main sample (Table 3, Panel A)
  >>> also re-balanced the original twelve (e.g. the China/Japan counts changed),
  >>> set TARGET_COUNTS and the candidate-list lengths below to match the N column
  >>> of Table 3, Panel A exactly, so the small-cap sample mirrors the headline one.

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
    pip install yfinance pandas numpy statsmodels scipy

--------------------------------------------------------------------------------
HOW TO USE
--------------------------------------------------------------------------------
1. Run the script once.  It first VALIDATES the candidate small-cap universe:
   it downloads each ticker, checks data coverage around the event, and prints
   the market capitalisation so you can confirm each firm is genuinely small-cap
   and headquartered in the stated country.
2. The tickers in FIRMS below are CANDIDATES.  Replace any that fail validation
   or are not small-cap.  The script will not proceed to the regressions until
   every country reaches its TARGET_COUNTS with valid firms.
3. Once the universe is complete, it runs the full event study and regressions
   and writes:
        smallcap_firm_level.csv     (one row per firm: CAR, controls, etc.)
        smallcap_descriptives.csv   (Panel A + Panel B, mirroring Table 3)
        smallcap_regressions.csv    (the four nested models, mirroring Table 9)

NOTE ON HONESTY: do not report any number this script produces without first
confirming the validation report looks sensible (coverage, market caps, signs).
================================================================================
"""

import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# 1. PARAMETERS                                                      #
# ------------------------------------------------------------------ #
EVENT_DATE      = pd.Timestamp("2025-04-02")   # t = 0
EST_START, EST_END = -250, -11                 # estimation window (trading-day offsets)
EVT_START, EVT_END = -1, +1                    # event window
DOWNLOAD_START  = "2023-01-01"
DOWNLOAD_END    = "2025-04-30"
USE_DIMSON      = True     # True -> Dimson (1979) lead/lag beta; False -> plain market model

# Unweighted mean tariff across the 16 sample countries is computed below (= 23.1),
# matching the thesis definition of the demeaned tariff shock.

# ------------------------------------------------------------------ #
# 2. COUNTRY-LEVEL DATA (identical to the main thesis, Table 3)      #
# ------------------------------------------------------------------ #
# Reciprocal tariff rate (%), from the White House / USTR schedule of 2 April 2025.
COUNTRY_TARIFF = {
    "Thailand": 36, "China": 34, "Indonesia": 32, "Taiwan": 32,
    "Switzerland": 31, "South Africa": 30, "India": 26, "South Korea": 25,
    "Japan": 24, "France": 20, "Germany": 20, "Netherlands": 20,
    "Australia": 10, "Brazil": 10, "Singapore": 10, "United Kingdom": 10,
}
# Financial development FDc = credit to private non-financial sector / GDP
# (BIS, Q4 2023; Taiwan from CBC Financial Stability Report).
COUNTRY_FD = {
    "Thailand": 1.776, "China": 1.930, "Indonesia": 0.400, "Taiwan": 1.661,
    "Switzerland": 2.536, "South Africa": 0.673, "India": 0.938, "South Korea": 2.059,
    "Japan": 1.749, "France": 2.166, "Germany": 1.396, "Netherlands": 2.853,
    "Australia": 1.686, "Brazil": 0.855, "Singapore": 1.726, "United Kingdom": 1.398,
}
# Primary local equity index for each country (Yahoo Finance ticker).
INDEX_BY_COUNTRY = {
    "China": "000300.SS",        # CSI 300 (A-shares); HK-listed firms override to ^HSI
    "Taiwan": "^TWII",           # TAIEX
    "Japan": "^N225",            # Nikkei 225
    "South Korea": "^KS11",      # KOSPI
    "India": "^NSEI",            # Nifty 50
    "Germany": "^GDAXI",         # DAX
    "France": "^FCHI",           # CAC 40
    "United Kingdom": "^FTSE",   # FTSE 100
    "Australia": "^AXJO",        # S&P/ASX 200
    "Brazil": "^BVSP",           # Ibovespa
    "Switzerland": "^SSMI",      # SMI
    "Singapore": "^STI",         # STI
    # ----- four new countries (VERIFY the index tickers below) -----
    "Thailand": "^SET.BK",       # SET Index   (verify on Yahoo: "^SET.BK")
    "Indonesia": "^JKSE",        # Jakarta Composite
    "South Africa": "^J203.JO",  # FTSE/JSE All Share (verify: "^J203.JO" / "^JN0U.JO")
    "Netherlands": "^AEX",       # AEX
}
# Target number of SMALL-CAP firms per country = the large-cap sample composition.
# >>> SET THESE TO MATCH THE N COLUMN OF TABLE 3, PANEL A. <<<
TARGET_COUNTS = {
    # original twelve (unchanged from the validated 12-country run)
    "China": 15, "Japan": 15, "United Kingdom": 10, "Taiwan": 8, "South Korea": 8,
    "India": 7, "Germany": 7, "France": 7, "Australia": 5, "Brazil": 5,
    "Switzerland": 4, "Singapore": 4,
    # four new countries (placeholders -- adjust to Table 3 Panel A; these sum to 19
    # so that 95 + 19 = 114)
    "Thailand": 5, "Indonesia": 5, "South Africa": 5, "Netherlands": 4,
}  # total = 114

# ------------------------------------------------------------------ #
# 3. CANDIDATE SMALL-CAP UNIVERSE  --  VERIFY / REPLACE BEFORE TRUST  #
# ------------------------------------------------------------------ #
# Each entry: (ticker, country[, index_override]).
# These are STARTING CANDIDATES selected as smaller listed firms in each market.
# You MUST confirm via the validation report that each one (a) downloads, (b) has
# full coverage of the [-250, +1] window, and (c) is genuinely small-cap. Swap out
# any that fail. The script enforces the per-country counts before running models.
FIRMS = [
    # --- China (15): A-shares on .SS/.SZ; add HK names with index override "^HSI"
    ("002030.SZ", "China"), ("002041.SZ", "China"), ("002146.SZ", "China"),
    ("002311.SZ", "China"), ("002384.SZ", "China"), ("002463.SZ", "China"),
    ("002507.SZ", "China"), ("002572.SZ", "China"), ("002818.SZ", "China"),
    ("603129.SS", "China"), ("603198.SS", "China"), ("603517.SS", "China"),
    ("603605.SS", "China"), ("603866.SS", "China"), ("603899.SS", "China"),
    # --- Japan (15): .T
    ("3097.T", "Japan"), ("3231.T", "Japan"), ("3543.T", "Japan"),
    ("4544.T", "Japan"), ("3656.T", "Japan"), ("3835.T", "Japan"),
    ("3923.T", "Japan"), ("4275.T", "Japan"), ("3994.T", "Japan"),
    ("6191.T", "Japan"), ("6232.T", "Japan"), ("3911.T", "Japan"),
    ("3962.T", "Japan"), ("3088.T", "Japan"), ("4475.T", "Japan"),
    # --- United Kingdom (10): .L
    ("CCC.L", "United Kingdom"), ("HFD.L", "United Kingdom"), ("MCB.L", "United Kingdom"),
    ("VANL.L", "United Kingdom"), ("TET.L", "United Kingdom"), ("CRST.L", "United Kingdom"),
    ("HSW.L", "United Kingdom"), ("GAW.L", "United Kingdom"), ("RWS.L", "United Kingdom"),
    ("VTY.L", "United Kingdom"),
    # --- Taiwan (8): .TW
    ("3036.TW", "Taiwan"), ("6147.TWO", "Taiwan"), ("8454.TW", "Taiwan"),
    ("5269.TW", "Taiwan"), ("3293.TWO", "Taiwan"), ("9914.TW", "Taiwan"),
    ("9921.TW", "Taiwan"), ("2231.TW", "Taiwan"),
    # --- South Korea (8): .KS / .KQ
    ("214150.KQ", "South Korea"), ("095610.KQ", "South Korea"), ("058470.KQ", "South Korea"),
    ("078600.KQ", "South Korea"), ("122870.KQ", "South Korea"), ("131970.KQ", "South Korea"),
    ("141080.KQ", "South Korea"), ("145020.KQ", "South Korea"),
    # --- India (7): .NS
    ("CAMS.NS", "India"), ("RADICO.NS", "India"), ("FINEORG.NS", "India"),
    ("KAJARIACER.NS", "India"), ("CDSL.NS", "India"), ("BLUESTARCO.NS", "India"),
    ("HAPPSTMNDS.NS", "India"),
    # --- Germany (7): .DE
    ("S92.DE", "Germany"), ("COK.DE", "Germany"), ("DMP.DE", "Germany"),
    ("NDX1.DE", "Germany"), ("VBK.DE", "Germany"), ("KWS.DE", "Germany"),
    ("ELG.DE", "Germany"),
    # --- France (7): .PA
    ("SOI.PA", "France"), ("MAU.PA", "France"), ("VLA.PA", "France"),
    ("ALTA.PA", "France"), ("GNFT.PA", "France"), ("AKW.PA", "France"),
    ("ICAD.PA", "France"),
    # --- Australia (5): .AX
    ("NWH.AX", "Australia"), ("CCP.AX", "Australia"), ("LOV.AX", "Australia"),
    ("BAP.AX", "Australia"), ("SGM.AX", "Australia"),
    # --- Brazil (5): .SA
    ("MOVI3.SA", "Brazil"), ("TUPY3.SA", "Brazil"), ("POMO4.SA", "Brazil"),
    ("LEVE3.SA", "Brazil"), ("GRND3.SA", "Brazil"),
    # --- Switzerland (4): .SW
    ("BUCN.SW", "Switzerland"), ("BCVN.SW", "Switzerland"), ("CLTN.SW", "Switzerland"),
    ("VBSN.SW", "Switzerland"),
    # --- Singapore (4): .SI
    ("AWX.SI", "Singapore"), ("BTOU.SI", "Singapore"), ("OYY.SI", "Singapore"),
    ("BSL.SI", "Singapore"),

    # ============================================================== #
    # FOUR NEW COUNTRIES -- UNVERIFIED PLACEHOLDERS, CONFIRM EACH ONE #
    # (check the coverage report: download OK, full window, small-cap) #
    # ============================================================== #
    # --- Thailand (5): .BK
    ("COM7.BK", "Thailand"), ("SAWAD.BK", "Thailand"), ("MTC.BK", "Thailand"),
    ("BCH.BK", "Thailand"), ("SPALI.BK", "Thailand"),
    # --- Indonesia (5): .JK
    ("MAPI.JK", "Indonesia"), ("ACES.JK", "Indonesia"), ("ERAA.JK", "Indonesia"),
    ("SCMA.JK", "Indonesia"), ("PWON.JK", "Indonesia"),
    # --- South Africa (5): .JO  (prices in ZA cents; returns unaffected)
    ("KAP.JO", "South Africa"), ("MRP.JO", "South Africa"), ("AVI.JO", "South Africa"),
    ("ARI.JO", "South Africa"), ("HMN.JO", "South Africa"),
    # --- Netherlands (4): .AS
    ("NEDAP.AS", "Netherlands"), ("ACOMO.AS", "Netherlands"), ("HEIJM.AS", "Netherlands"),
    ("KENDR.AS", "Netherlands"),
]

# Clean up the FIRMS list to (ticker, country, index) triples.
def _resolve(entry):
    if len(entry) == 3:
        return entry[0], entry[1], entry[2]
    t, c = entry[0], entry[1]
    return t, c, INDEX_BY_COUNTRY[c]

UNIVERSE = [_resolve(e) for e in FIRMS]

# ------------------------------------------------------------------ #
# 4. PRICE / RETURN HELPERS                                          #
# ------------------------------------------------------------------ #
def fetch_prices(ticker):
    """Return a Series of adjusted close prices, or None on failure."""
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=DOWNLOAD_START, end=DOWNLOAD_END,
                             auto_adjust=True, progress=False, threads=False)
            if df is None or df.empty:
                return None
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.name = ticker
            return s.dropna()
        except Exception:
            time.sleep(1.5)
    return None


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def event_study(firm_ret, mkt_ret):
    """Market-model (optionally Dimson) CAR[-1,+1] and estimation-window volatility."""
    df = pd.concat([firm_ret, mkt_ret], axis=1, keys=["r", "rm"]).dropna()
    if df.empty:
        return None
    dates = df.index
    pos = dates.searchsorted(EVENT_DATE, side="right") - 1   # last trading day <= event
    if pos < 0:
        return None
    est_lo, est_hi = pos + EST_START, pos + EST_END
    evt_lo, evt_hi = pos + EVT_START, pos + EVT_END
    need_lo = est_lo - (1 if USE_DIMSON else 0)
    if need_lo < 0 or evt_hi >= len(df):
        return None  # insufficient coverage around the event

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

    vol = est["r"].std() * 100.0
    ew  = df.iloc[evt_lo:evt_hi + 1]
    ar  = ew["r"] - (alpha + beta * ew["rm"])
    car = ar.sum() * 100.0
    return {"CAR": car, "alpha": alpha, "beta": beta,
            "vol": vol, "n_est": len(est)}


def get_financials(ticker):
    """Most recent annual fundamentals prior to the event, plus market cap."""
    out = {k: np.nan for k in
           ["total_assets", "total_debt", "book_equity", "net_income",
            "cash", "market_cap", "currency"]}
    try:
        tk = yf.Ticker(ticker)
        bs = tk.balance_sheet
        fin = tk.financials

        def pick(frame, *names):
            if frame is None or frame.empty:
                return np.nan
            cols = sorted(frame.columns)          # ascending dates
            col = cols[-1]                         # most recent annual column
            for nm in names:
                if nm in frame.index:
                    return float(frame.loc[nm, col])
            return np.nan

        out["total_assets"] = pick(bs, "Total Assets")
        out["total_debt"]   = pick(bs, "Total Debt", "Long Term Debt")
        out["book_equity"]  = pick(bs, "Stockholders Equity",
                                   "Total Stockholder Equity",
                                   "Common Stock Equity")
        out["cash"]         = pick(bs, "Cash And Cash Equivalents",
                                   "Cash Cash Equivalents And Short Term Investments")
        out["net_income"]   = pick(fin, "Net Income", "Net Income Common Stockholders")

        mcap = np.nan
        try:
            fi = tk.fast_info
            # yfinance FastInfo uses camelCase keys ("marketCap"), not "market_cap"
            for k in ("marketCap", "market_cap"):
                try:
                    v = fi[k]
                    if v:
                        mcap = float(v)
                        break
                except Exception:
                    pass
            if not (mcap == mcap) or mcap is None:   # still NaN -> shares * price
                try:
                    sh, px = fi["shares"], fi["lastPrice"]
                    if sh and px:
                        mcap = float(sh) * float(px)
                except Exception:
                    pass
            try:
                out["currency"] = fi["currency"]
            except Exception:
                out["currency"] = ""
        except Exception:
            pass
        if not (mcap == mcap):   # last resort: the slow .info call
            try:
                mcap = float(tk.info.get("marketCap", np.nan))
            except Exception:
                pass
        out["market_cap"] = mcap
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ #
# 5. VALIDATION + ASSEMBLY                                          #
# ------------------------------------------------------------------ #
def build_panel():
    # download all unique index series once
    idx_tickers = sorted({ix for _, _, ix in UNIVERSE})
    idx_ret = {}
    print("Downloading market indices ...")
    for ix in idx_tickers:
        p = fetch_prices(ix)
        idx_ret[ix] = log_returns(p) if p is not None else None
        print(f"  {ix:12s} {'ok' if idx_ret[ix] is not None else 'FAILED'}")

    rows, report = [], []
    print("\nValidating and processing firms ...")
    for ticker, country, ix in UNIVERSE:
        p = fetch_prices(ticker)
        if p is None:
            report.append((ticker, country, "NO PRICE DATA", np.nan))
            print(f"  {ticker:14s} {country:14s} NO PRICE DATA")
            continue
        if idx_ret.get(ix) is None:
            report.append((ticker, country, "INDEX MISSING", np.nan))
            continue

        es = event_study(log_returns(p), idx_ret[ix])
        fn = get_financials(ticker)
        if es is None:
            report.append((ticker, country, "INSUFFICIENT WINDOW", fn["market_cap"]))
            print(f"  {ticker:14s} {country:14s} INSUFFICIENT WINDOW "
                  f"(mcap={fn['market_cap']:.3g} {fn['currency']})")
            continue

        size = np.log(fn["total_assets"]) if fn["total_assets"] and fn["total_assets"] > 0 else np.nan
        lev  = (fn["total_debt"] / fn["total_assets"]
                if fn["total_assets"] and fn["total_debt"] is not None else np.nan)
        btm  = (fn["book_equity"] / fn["market_cap"]
                if fn["market_cap"] and fn["book_equity"] is not None else np.nan)
        roa  = (fn["net_income"] / fn["total_assets"]
                if fn["total_assets"] and fn["net_income"] is not None else np.nan)
        cash = (fn["cash"] / fn["total_assets"]
                if fn["total_assets"] and fn["cash"] is not None else np.nan)

        rows.append({
            "ticker": ticker, "country": country,
            "CAR": es["CAR"], "beta": es["beta"], "volatility": es["vol"],
            "ln_assets": size, "leverage": lev, "book_to_market": btm,
            "ROA": roa, "cash_assets": cash,
            "market_cap": fn["market_cap"], "currency": fn["currency"],
            "tariff": COUNTRY_TARIFF[country], "FDc": COUNTRY_FD[country],
        })
        report.append((ticker, country, "OK", fn["market_cap"]))
        print(f"  {ticker:14s} {country:14s} OK   CAR={es['CAR']:+6.2f}%  "
              f"beta={es['beta']:.2f}  mcap={fn['market_cap']:.3g} {fn['currency']}")
        time.sleep(0.3)  # be gentle with the API

    panel = pd.DataFrame(rows)

    # enforce per-country counts before allowing inference
    print("\n----- COVERAGE CHECK -----")
    ok = True
    for c, target in TARGET_COUNTS.items():
        have = int((panel["country"] == c).sum()) if not panel.empty else 0
        flag = "OK" if have >= target else "SHORT  <-- add/replace tickers"
        if have < target:
            ok = False
        print(f"  {c:16s} have {have:2d} / target {target:2d}   {flag}")
    return panel, ok


# ------------------------------------------------------------------ #
# 6. DESCRIPTIVES (mirrors Table 3)                                  #
# ------------------------------------------------------------------ #
def descriptives(panel):
    # aggregate reaction (Section 5.1 analogue)
    car = panel["CAR"].dropna()
    n = len(car)
    tstat = car.mean() / (car.std(ddof=1) / np.sqrt(n))
    print("\n========== AGGREGATE MARKET REACTION ==========")
    print(f"  N firms              : {n}")
    print(f"  Mean CAR[-1,+1]      : {car.mean():+.4f} %")
    print(f"  Median CAR[-1,+1]    : {car.median():+.4f} %")
    print(f"  t-statistic (H0=0)   : {tstat:+.3f}")
    print(f"  Share negative       : {100*(car<0).mean():.1f} %")

    # Panel A: by country
    pa = (panel.groupby("country")
          .agg(N=("CAR", "size"),
               Tariff=("tariff", "first"),
               FDc=("FDc", "first"),
               Mean_CAR=("CAR", "mean"),
               Median_CAR=("CAR", "median"))
          .reset_index())

    # Panel B: summary stats
    vars_ = ["CAR", "tariff", "FDc", "ln_assets", "leverage",
             "book_to_market", "ROA", "cash_assets", "volatility"]
    pb = []
    for v in vars_:
        s = panel[v].dropna()
        jb, jbp = stats.jarque_bera(s) if len(s) > 7 else (np.nan, np.nan)
        pb.append({"Variable": v, "N": len(s), "Mean": s.mean(),
                   "Median": s.median(), "Std": s.std(ddof=1),
                   "Min": s.min(), "Max": s.max(),
                   "Skew": stats.skew(s), "ExKurt": stats.kurtosis(s),
                   "JB": jb, "JB_p": jbp})
    pb = pd.DataFrame(pb)

    print("\n========== PANEL A: BY COUNTRY ==========")
    print(pa.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n========== PANEL B: SUMMARY STATISTICS ==========")
    print(pb.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    pa.to_csv("smallcap_descriptives_panelA.csv", index=False)
    pb.to_csv("smallcap_descriptives_panelB.csv", index=False)
    return pa, pb


# ------------------------------------------------------------------ #
# 7. CROSS-SECTIONAL REGRESSIONS (mirrors Table 9)                   #
# ------------------------------------------------------------------ #
def regressions(panel):
    df = panel.dropna(subset=["CAR"]).copy()
    mean_tariff = np.mean(list(COUNTRY_TARIFF.values()))   # unweighted 16-country mean = 23.1
    df["tariff_shock"] = df["tariff"] - mean_tariff
    df["fd_x_shock"]   = df["FDc"] * df["tariff_shock"]
    controls = ["ln_assets", "leverage", "book_to_market", "ROA", "cash_assets", "volatility"]

    specs = {
        "(1)": ["tariff_shock"],
        "(2)": ["tariff_shock"] + controls,
        "(3)": ["tariff_shock", "FDc", "fd_x_shock"],
        "(4)": ["tariff_shock", "FDc", "fd_x_shock"] + controls,
    }

    def stars(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    results = {}
    for name, regs in specs.items():
        d = df.dropna(subset=regs + ["CAR"]).copy()
        X = sm.add_constant(d[regs])
        model = sm.OLS(d["CAR"], X).fit(
            cov_type="cluster", cov_kwds={"groups": d["country"]})
        results[name] = (model, d.shape[0])

    # assemble a tidy comparison table
    order = ["tariff_shock", "FDc", "fd_x_shock"] + controls + ["const"]
    label = {"tariff_shock": "Tariff shock", "FDc": "FDc",
             "fd_x_shock": "FDc x Tariff shock", "ln_assets": "ln(Total assets)",
             "leverage": "Leverage", "book_to_market": "Book-to-market",
             "ROA": "ROA", "cash_assets": "Cash/Assets",
             "volatility": "Volatility", "const": "Constant"}
    rows = []
    for var in order:
        row = {"Variable": label[var]}
        for name, (m, _) in results.items():
            if var in m.params.index:
                row[name] = f"{m.params[var]:.4f}{stars(m.pvalues[var])}"
                row[name + "_se"] = f"({m.bse[var]:.4f})"
            else:
                row[name] = ""
                row[name + "_se"] = ""
        rows.append(row)
    tab = pd.DataFrame(rows)

    print("\n========== TABLE: CROSS-SECTIONAL REGRESSIONS (CAR[-1,+1] %) ==========")
    for _, r in tab.iterrows():
        print(f"{r['Variable']:>20s}  "
              + "  ".join(f"{r[c]:>12s}" for c in ['(1)', '(2)', '(3)', '(4)']))
        print(f"{'':>20s}  "
              + "  ".join(f"{r[c+'_se']:>12s}" for c in ['(1)', '(2)', '(3)', '(4)']))
    print("-" * 78)
    print(f"{'Observations':>20s}  "
          + "  ".join(f"{results[c][1]:>12d}" for c in ['(1)', '(2)', '(3)', '(4)']))
    print(f"{'R-squared':>20s}  "
          + "  ".join(f"{results[c][0].rsquared:>12.4f}" for c in ['(1)', '(2)', '(3)', '(4)']))
    print(f"{'Adj. R-squared':>20s}  "
          + "  ".join(f"{results[c][0].rsquared_adj:>12.4f}" for c in ['(1)', '(2)', '(3)', '(4)']))

    tab.to_csv("smallcap_regressions.csv", index=False)
    return results, tab


# ------------------------------------------------------------------ #
# 7b. WILD CLUSTER BOOTSTRAP  (Cameron, Gelbach & Miller 2008)        #
#     Restricted (null-imposed) wild bootstrap with Rademacher weights #
#     -> one-sided p-values for the coefficients tested in H1 and H2.  #
# ------------------------------------------------------------------ #
def _design(d, regs):
    return np.column_stack([np.ones(len(d))] + [d[r].values.astype(float) for r in regs])


def _cluster_t(Xf, y, codes, n_groups, XtX_inv, P, k):
    """Coefficient k: cluster-robust t-stat with the CGM small-sample correction."""
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
    se = np.sqrt(V[k, k])
    return beta[k] / se, beta[k], se


def wild_cluster_bootstrap(df, regs, coef, group_col="country", B=9999,
                           tail="left", seed=12345):
    """One-sided wild cluster bootstrap p-value for `coef` in the full model.
       tail='left'  -> H1 (expect coef < 0);  tail='right' -> H2 (expect coef > 0)."""
    d = df.dropna(subset=regs + ["CAR"]).copy()
    y = d["CAR"].values.astype(float)
    Xf = _design(d, regs)
    names = ["const"] + regs
    k = names.index(coef)
    cats = d[group_col].astype("category")
    codes = cats.cat.codes.values
    n_groups = len(cats.cat.categories)

    XtX_inv = np.linalg.inv(Xf.T @ Xf)
    P = XtX_inv @ Xf.T
    t_orig, b_orig, se_orig = _cluster_t(Xf, y, codes, n_groups, XtX_inv, P, k)

    # restricted fit: impose the null by dropping the tested regressor
    r_regs = [r for r in regs if r != coef]
    Xr = _design(d, r_regs)
    beta_r = np.linalg.inv(Xr.T @ Xr) @ Xr.T @ y
    yhat_r = Xr @ beta_r
    u_r = y - yhat_r

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(B):
        w = (rng.integers(0, 2, size=n_groups) * 2 - 1)[codes].astype(float)  # Rademacher
        y_star = yhat_r + w * u_r
        t_b, _, _ = _cluster_t(Xf, y_star, codes, n_groups, XtX_inv, P, k)
        if (t_b >= t_orig) if tail == "right" else (t_b <= t_orig):
            count += 1
    return t_orig, b_orig, se_orig, count / B


def run_bootstrap(panel, B=9999):
    d = panel.dropna(subset=["CAR"]).copy()
    d["tariff_shock"] = d["tariff"] - np.mean(list(COUNTRY_TARIFF.values()))
    d["fd_x_shock"] = d["FDc"] * d["tariff_shock"]
    controls = ["ln_assets", "leverage", "book_to_market", "ROA", "cash_assets", "volatility"]
    full = ["tariff_shock", "FDc", "fd_x_shock"] + controls

    print("\n" + "=" * 70)
    print(f"WILD CLUSTER BOOTSTRAP  (full model, {B} reps, Rademacher, null imposed)")
    print("=" * 70)
    t3, b3, se3, p3 = wild_cluster_bootstrap(d, full, "fd_x_shock", tail="right", B=B)
    print(f"  H2  interaction (FDc x Tariff):  coef={b3:+.4f}  t={t3:+.3f}"
          f"   one-sided bootstrap p = {p3:.4f}")
    t1, b1, se1, p1 = wild_cluster_bootstrap(d, full, "tariff_shock", tail="left", B=B)
    print(f"  H1  tariff shock:                coef={b1:+.4f}  t={t1:+.3f}"
          f"   one-sided bootstrap p = {p1:.4f}")
    print("=" * 70)
    pd.DataFrame([
        {"coef": "FDc x Tariff shock (H2)", "estimate": b3, "t_clustered": t3,
         "tail": "right", "bootstrap_p_1sided": p3},
        {"coef": "Tariff shock (H1)", "estimate": b1, "t_clustered": t1,
         "tail": "left", "bootstrap_p_1sided": p1},
    ]).to_csv("smallcap_bootstrap.csv", index=False)
    print("Saved -> smallcap_bootstrap.csv")


# ------------------------------------------------------------------ #
# 8. MAIN                                                            #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Small-cap robustness replication (16-country)  |  Dimson correction:",
          "ON" if USE_DIMSON else "OFF")
    print(f"Unweighted 16-country mean tariff = "
          f"{np.mean(list(COUNTRY_TARIFF.values())):.1f}%  (demeaning constant)")
    panel, complete = build_panel()
    if not panel.empty:
        panel.to_csv("smallcap_firm_level.csv", index=False)

    if not complete:
        print("\n*** Universe incomplete. Replace the flagged tickers so every "
              "country reaches its target count, then re-run. Regressions skipped. ***")
    else:
        descriptives(panel)
        regressions(panel)
        run_bootstrap(panel)
        print("\nDone. Outputs written to smallcap_*.csv")
