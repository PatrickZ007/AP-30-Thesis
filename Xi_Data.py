"""
Fetch firm-level control variables from Yahoo Finance
- Financial statements: FY2023 (year-end 2023)
- Volatility: estimation window [-250, -11] relative to April 2, 2025
- Output: firm_controls.xlsx

Run: pip3 install yfinance pandas openpyxl
     python3 fetch_firm_data.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. DEFINE SAMPLE
# ============================================================
firms = [
    # China
    ("000333.SZ", "Midea Group", "China"),
    ("2359.HK", "WuXi AppTec", "China"),
    ("1211.HK", "BYD Company", "China"),
    ("2899.HK", "Zijin Mining", "China"),
    ("300750.SZ", "CATL", "China"),
    ("9988.HK", "Alibaba Group", "China"),
    ("3968.HK", "China Merchants Bank", "China"),
    ("1398.HK", "ICBC", "China"),
    ("0883.HK", "CNOOC", "China"),
    ("0939.HK", "China Construction Bank", "China"),
    ("600519.SS", "Kweichow Moutai", "China"),
    ("0700.HK", "Tencent Holdings", "China"),
    ("2318.HK", "Ping An Insurance", "China"),
    ("1109.HK", "China Resources Land", "China"),
    ("0788.HK", "China Tower", "China"),
    # Taiwan
    ("2308.TW", "Delta Electronics", "Taiwan"),
    ("2317.TW", "Hon Hai (Foxconn)", "Taiwan"),
    ("2882.TW", "Cathay Financial", "Taiwan"),
    ("3711.TW", "ASE Technology", "Taiwan"),
    ("2330.TW", "TSMC", "Taiwan"),
    ("2881.TW", "Fubon Financial", "Taiwan"),
    ("2454.TW", "MediaTek", "Taiwan"),
    ("1216.TW", "Uni-President Enterprises", "Taiwan"),
    # Switzerland
    ("UBSG.SW", "UBS Group", "Switzerland"),
    ("ABBN.SW", "ABB", "Switzerland"),
    ("NOVN.SW", "Novartis", "Switzerland"),
    ("NESN.SW", "Nestlé", "Switzerland"),
    # India
    ("TCS.NS", "Tata Consultancy", "India"),
    ("INFY.NS", "Infosys", "India"),
    ("RELIANCE.NS", "Reliance Industries", "India"),
    ("BHARTIARTL.NS", "Bharti Airtel", "India"),
    ("HDFCBANK.NS", "HDFC Bank", "India"),
    ("ICICIBANK.NS", "ICICI Bank", "India"),
    ("SUNPHARMA.NS", "Sun Pharmaceutical", "India"),
    # South Korea
    ("373220.KS", "LG Energy Solution", "South Korea"),
    ("105560.KS", "KB Financial Group", "South Korea"),
    ("005490.KS", "POSCO Holdings", "South Korea"),
    ("005380.KS", "Hyundai Motor", "South Korea"),
    ("005930.KS", "Samsung Electronics", "South Korea"),
    ("000660.KS", "SK Hynix", "South Korea"),
    ("035420.KS", "Naver", "South Korea"),
    ("207940.KS", "Samsung Biologics", "South Korea"),
    # Japan
    ("8316.T", "Sumitomo Mitsui Financial", "Japan"),
    ("8306.T", "Mitsubishi UFJ Financial", "Japan"),
    ("6758.T", "Sony Group", "Japan"),
    ("3382.T", "Seven & i Holdings", "Japan"),
    ("6501.T", "Hitachi", "Japan"),
    ("9433.T", "KDDI", "Japan"),
    ("6861.T", "Keyence", "Japan"),
    ("8766.T", "Tokio Marine Holdings", "Japan"),
    ("7203.T", "Toyota Motor", "Japan"),
    ("4063.T", "Shin-Etsu Chemical", "Japan"),
    ("7974.T", "Nintendo", "Japan"),
    ("8035.T", "Tokyo Electron", "Japan"),
    ("9983.T", "Fast Retailing", "Japan"),
    ("4568.T", "Daiichi Sankyo", "Japan"),
    ("6098.T", "Recruit Holdings", "Japan"),
    # Germany
    ("SIE.DE", "Siemens", "Germany"),
    ("BAS.DE", "BASF", "Germany"),
    ("MBG.DE", "Mercedes-Benz Group", "Germany"),
    ("BMW.DE", "BMW", "Germany"),
    ("DTE.DE", "Deutsche Telekom", "Germany"),
    ("SAP.DE", "SAP", "Germany"),
    ("ALV.DE", "Allianz", "Germany"),
    # France
    ("TTE.PA", "TotalEnergies", "France"),
    ("BNP.PA", "BNP Paribas", "France"),
    ("SAN.PA", "Sanofi", "France"),
    ("MC.PA", "LVMH", "France"),
    ("SU.PA", "Schneider Electric", "France"),
    ("AIR.PA", "Airbus", "France"),
    ("OR.PA", "L'Oréal", "France"),
    # Singapore
    ("SE", "Sea Limited", "Singapore"),
    ("D05.SI", "DBS Group Holdings", "Singapore"),
    ("F34.SI", "Wilmar International", "Singapore"),
    ("Z74.SI", "Singapore Telecom", "Singapore"),
    # United Kingdom
    ("HSBA.L", "HSBC Holdings", "United Kingdom"),
    ("BP.L", "BP", "United Kingdom"),
    ("SHEL.L", "Shell", "United Kingdom"),
    ("RIO.L", "Rio Tinto", "United Kingdom"),
    ("RR.L", "Rolls-Royce Holdings", "United Kingdom"),
    ("AZN.L", "AstraZeneca", "United Kingdom"),
    ("DGE.L", "Diageo", "United Kingdom"),
    ("LSEG.L", "London Stock Exchange", "United Kingdom"),
    ("ULVR.L", "Unilever", "United Kingdom"),
    ("BA.L", "BAE Systems", "United Kingdom"),
    # Brazil
    ("PETR4.SA", "Petrobras", "Brazil"),
    ("VALE3.SA", "Vale", "Brazil"),
    ("WEGE3.SA", "WEG", "Brazil"),
    ("ITUB4.SA", "Itaú Unibanco", "Brazil"),
    ("ABEV3.SA", "Ambev", "Brazil"),
    # Australia
    ("BHP.AX", "BHP Group", "Australia"),
    ("WDS.AX", "Woodside Energy", "Australia"),
    ("WES.AX", "Wesfarmers", "Australia"),
    ("CBA.AX", "Commonwealth Bank", "Australia"),
    ("CSL.AX", "CSL Limited", "Australia"),
]

# ============================================================
# 2. DATE PARAMETERS
# ============================================================
event_date = datetime(2025, 4, 2)

# Estimation window [-250, -11]: need price data from ~April 2024 to ~March 18, 2025
# Download extra buffer for trading day alignment
price_start = "2024-03-01"
price_end = "2025-04-10"

# FY2023 financial data: year ending Dec 31, 2023
# (Japanese firms with March FY: year ending March 31, 2024)
target_fy = "2023"

# Market cap reference date: Dec 31, 2023
mktcap_date = "2023-12-31"

# ============================================================
# 3. FETCH DATA
# ============================================================
results = []
errors = []

print(f"Fetching data for {len(firms)} firms...")
print("=" * 70)

for i, (ticker, name, country) in enumerate(firms):
    print(f"[{i+1}/{len(firms)}] {ticker} ({name})...", end=" ")

    try:
        stock = yf.Ticker(ticker)

        # ----- Financial statements (FY2023) -----
        bs = stock.balance_sheet
        inc = stock.financials

        if bs is None or bs.empty:
            print("NO FINANCIALS")
            errors.append((ticker, name, "No balance sheet"))
            continue

        # Find the column closest to FY2023 (Dec 2023 or March 2024 for Japan)
        # Balance sheet columns are datetime index
        fy_col = None
        for col in bs.columns:
            col_year = col.year
            col_month = col.month
            # Accept Dec 2023 or any date in 2023, or March 2024 (Japan FY)
            if col_year == 2023 and col_month >= 9:
                fy_col = col
                break
            elif col_year == 2024 and col_month <= 6:
                fy_col = col
                break
        
        if fy_col is None and len(bs.columns) > 0:
            # Fallback: use the most recent available
            fy_col = bs.columns[0]

        # Extract balance sheet items
        def get_item(df, keys, col):
            for key in keys:
                if key in df.index:
                    val = df.loc[key, col]
                    if pd.notna(val):
                        return float(val)
            return np.nan

        total_assets = get_item(bs, ["Total Assets", "TotalAssets"], fy_col)
        total_debt = get_item(bs, [
            "Total Debt", "TotalDebt",
            "Long Term Debt", "LongTermDebt",
            "Net Debt", "Total Non Current Liabilities Net Minority Interest"
        ], fy_col)
        book_equity = get_item(bs, [
            "Stockholders Equity", "StockholdersEquity",
            "Total Equity Gross Minority Interest",
            "Common Stock Equity", "Ordinary Shares Number"
        ], fy_col)
        cash = get_item(bs, [
            "Cash And Cash Equivalents", "CashAndCashEquivalents",
            "Cash Cash Equivalents And Short Term Investments"
        ], fy_col)

        # Net income from income statement
        net_income = np.nan
        if inc is not None and not inc.empty:
            inc_col = None
            for col in inc.columns:
                if col.year == 2023 and col.month >= 9:
                    inc_col = col
                    break
                elif col.year == 2024 and col.month <= 6:
                    inc_col = col
                    break
            if inc_col is None and len(inc.columns) > 0:
                inc_col = inc.columns[0]
            if inc_col is not None:
                net_income = get_item(inc, [
                    "Net Income", "NetIncome",
                    "Net Income Common Stockholders"
                ], inc_col)

        # ----- Market cap (end of 2023) -----
        try:
            price_hist = stock.history(start="2023-12-20", end="2024-01-10")
            if not price_hist.empty:
                close_price = price_hist["Close"].iloc[-1]
                shares = stock.info.get("sharesOutstanding", np.nan)
                if pd.notna(shares):
                    market_cap = close_price * shares
                else:
                    market_cap = stock.info.get("marketCap", np.nan)
            else:
                market_cap = stock.info.get("marketCap", np.nan)
        except:
            market_cap = stock.info.get("marketCap", np.nan)

        # ----- Volatility (estimation window) -----
        try:
            hist = stock.history(start=price_start, end=price_end, auto_adjust=True)
            if not hist.empty and len(hist) > 50:
                hist["log_ret"] = np.log(hist["Close"] / hist["Close"].shift(1))
                hist = hist.dropna(subset=["log_ret"])

                # Find event date index
                hist.index = hist.index.tz_localize(None)
                # Get trading days before event
                pre_event = hist[hist.index < event_date].copy()

                if len(pre_event) >= 250:
                    est_window = pre_event.iloc[-250:-11]
                elif len(pre_event) >= 100:
                    est_window = pre_event.iloc[:-11]
                else:
                    est_window = pre_event

                volatility = est_window["log_ret"].std() * 100  # in percentage
            else:
                volatility = np.nan
        except:
            volatility = np.nan

        # ----- Compute ratios -----
        ln_assets = np.log(total_assets) if pd.notna(total_assets) and total_assets > 0 else np.nan
        leverage = total_debt / total_assets if pd.notna(total_debt) and pd.notna(total_assets) and total_assets > 0 else np.nan
        bm = book_equity / market_cap if pd.notna(book_equity) and pd.notna(market_cap) and market_cap > 0 else np.nan
        roa = net_income / total_assets if pd.notna(net_income) and pd.notna(total_assets) and total_assets > 0 else np.nan
        cash_ratio = cash / total_assets if pd.notna(cash) and pd.notna(total_assets) and total_assets > 0 else np.nan

        results.append({
            "Ticker": ticker,
            "Company": name,
            "Country": country,
            "FY_Date": fy_col.strftime("%Y-%m-%d") if fy_col is not None else "",
            "Total_Assets": total_assets,
            "Total_Debt": total_debt,
            "Book_Equity": book_equity,
            "Cash": cash,
            "Net_Income": net_income,
            "Market_Cap": market_cap,
            "ln_Assets": ln_assets,
            "Leverage": leverage,
            "BM": bm,
            "ROA": roa,
            "Cash_Assets": cash_ratio,
            "Volatility": volatility,
        })

        print(f"OK (FY: {fy_col.strftime('%Y-%m') if fy_col else 'N/A'})")

    except Exception as e:
        print(f"ERROR: {e}")
        errors.append((ticker, name, str(e)))

# ============================================================
# 4. EXPORT
# ============================================================
df = pd.DataFrame(results)

print(f"\n{'='*70}")
print(f"Successfully fetched: {len(df)}/{len(firms)} firms")
print(f"Errors: {len(errors)}")

if errors:
    print("\nFailed tickers:")
    for t, n, e in errors:
        print(f"  {t} ({n}): {e}")

# Check missing values
print(f"\nMissing values:")
for col in ["ln_Assets", "Leverage", "BM", "ROA", "Cash_Assets", "Volatility"]:
    missing = df[col].isna().sum()
    print(f"  {col}: {missing}/{len(df)}")

# Save
with pd.ExcelWriter("firm_controls_FY2023.xlsx") as writer:
    # Main output
    output_cols = ["Ticker", "Company", "Country", "FY_Date",
                   "ln_Assets", "Leverage", "BM", "ROA", "Cash_Assets", "Volatility",
                   "Market_Cap"]
    df[output_cols].to_excel(writer, sheet_name="Controls", index=False)

    # Raw data for verification
    df.to_excel(writer, sheet_name="Raw_Data", index=False)

    # Errors log
    if errors:
        pd.DataFrame(errors, columns=["Ticker", "Company", "Error"]).to_excel(
            writer, sheet_name="Errors", index=False
        )

print(f"\nSaved to firm_controls_FY2023.xlsx")
print("\nPreview:")
print(df[["Ticker", "Company", "Country", "ln_Assets", "Leverage", "BM", "ROA", "Cash_Assets", "Volatility"]].to_string(index=False))
