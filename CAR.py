"""
====================================================================
 Event Study CAR Calculation — Embedded Ticker Version
====================================================================
Purpose:
  Calculate firm-level abnormal returns (AR) and cumulative abnormal
  returns (CAR[-1,+1]) around the event date 2025-04-02.

Model:
  R_i,t = alpha_i + beta_i * R_local-market,t + epsilon_i,t
  AR_i,t = R_i,t - E[R_i,t]
  CAR_i[-1,+1] = sum AR_i,t from t=-1 to t=+1

Default settings:
  Event date:        2025-04-02
  Estimation window: [-250, -11]
  Event window:      [-1, +1]
  Returns:           daily log returns from adjusted close prices
  Price source:      Yahoo Finance via yfinance

Install requirements:
  python3 -m pip install yfinance pandas numpy statsmodels scipy

Run examples:
  cd ~/Desktop
  python3 get_car_event_study_embedded.py

Notes: 
  The final country-level sample excludes Malaysia, Italy and Spain

  # Optional: save terminal results to CSV files
  python3 get_car_event_study_embedded.py --save-csv

====================================================================
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# 1. DEFAULT CONFIGURATION
# ============================================================

EVENT_DATE = "2025-04-02"
DOWNLOAD_START = "2023-06-01"
DOWNLOAD_END = "2025-04-11"

EST_WINDOW_START = -250
EST_WINDOW_END = -11
EVT_WINDOW_START = -1
EVT_WINDOW_END = 1
MIN_EST_DAYS = 192  # 80% of the 240-trading-day estimation window

DEFAULT_OUTPUT_DIR = "output_car"

# If a market index in FIRM_LIST contains "verify", this fallback is used.
# Some frontier/emerging-market Yahoo index tickers may need manual adjustment
# depending on Yahoo Finance coverage at runtime.
DEFAULT_MARKET_INDEX = {
    "China": "000300.SS",
    "Taiwan": "^TWII",
    "Vietnam": "^VNINDEX",
    "Thailand": "^SET.BK",
    "India": "^NSEI",
    "Indonesia": "^JKSE",
    "Malaysia": "^KLSE",
    "South Korea": "^KS11",
    "Japan": "^N225",
    "France": "^FCHI",
    "Germany": "^GDAXI",
    "Switzerland": "^SSMI",
    "Australia": "^AXJO",
    "Brazil": "^BVSP",
    "United Kingdom": "^FTSE",
    "Singapore": "^STI",
    "Italy": "FTSEMIB.MI",
    "Spain": "^IBEX",
    "Netherlands": "^AEX",
    "South Africa": "^J203.JO",
}


# ============================================================
# 2. EMBEDDED COMPANY TICKER LIST
# ============================================================

# Format:
#   country       : firm's headquarters country
#   company       : company name
#   ticker        : Yahoo Finance ticker for the company
#   market_index  : Yahoo Finance ticker for the local market index

FIRM_LIST = [
    {
        "country": "China",
        "company": "Kweichow Moutai",
        "ticker": "600519.SS",
        "market_index": "000300.SS"
    },
    {
        "country": "China",
        "company": "Tencent Holdings",
        "ticker": "0700.HK",
        "market_index": "^HSI"
    },
    {
        "country": "China",
        "company": "Alibaba Group",
        "ticker": "9988.HK",
        "market_index": "^HSI"
    },
    {
        "country": "China",
        "company": "BYD Company",
        "ticker": "1211.HK",
        "market_index": "^HSI"
    },
    {
        "country": "China",
        "company": "CATL",
        "ticker": "300750.SZ",
        "market_index": "000300.SS"
    },
    {
        "country": "China",
        "company": "Midea Group",
        "ticker": "000333.SZ",
        "market_index": "000300.SS"
    },
    {
        "country": "China",
        "company": "Zijin Mining",
        "ticker": "2899.HK",
        "market_index": "^HSI"
    },
    {
        "country": "China",
        "company": "CNOOC",
        "ticker": "0883.HK",
        "market_index": "^HSI"
    },
    {
        "country": "Taiwan",
        "company": "TSMC",
        "ticker": "2330.TW",
        "market_index": "^TWII"
    },
    {
        "country": "Taiwan",
        "company": "Hon Hai (Foxconn)",
        "ticker": "2317.TW",
        "market_index": "^TWII"
    },
    {
        "country": "Taiwan",
        "company": "MediaTek",
        "ticker": "2454.TW",
        "market_index": "^TWII"
    },
    {
        "country": "Taiwan",
        "company": "Delta Electronics",
        "ticker": "2308.TW",
        "market_index": "^TWII"
    },
    {
        "country": "Taiwan",
        "company": "ASE Technology",
        "ticker": "3711.TW",
        "market_index": "^TWII"
    },
    {
        "country": "Taiwan",
        "company": "Uni-President Enterprises",
        "ticker": "1216.TW",
        "market_index": "^TWII"
    },
    {
        "country": "Taiwan",
        "company": "Cathay Financial",
        "ticker": "2882.TW",
        "market_index": "^TWII"
    },
    {
        "country": "Vietnam",
        "company": "Vietcombank",
        "ticker": "VCB.VN",
        "market_index": "VNINDEX / verify"
    },
    {
        "country": "Vietnam",
        "company": "Vingroup",
        "ticker": "VIC.VN",
        "market_index": "VNINDEX / verify"
    },
    {
        "country": "Vietnam",
        "company": "Vinhomes",
        "ticker": "VHM.VN",
        "market_index": "VNINDEX / verify"
    },
    {
        "country": "Vietnam",
        "company": "Vinamilk",
        "ticker": "VNM.VN",
        "market_index": "VNINDEX / verify"
    },
    {
        "country": "Vietnam",
        "company": "FPT Corporation",
        "ticker": "FPT.VN",
        "market_index": "VNINDEX / verify"
    },
    {
        "country": "Vietnam",
        "company": "Hoa Phat Group",
        "ticker": "HPG.VN",
        "market_index": "VNINDEX / verify"
    },
    {
        "country": "Vietnam",
        "company": "Masan Group",
        "ticker": "MSN.VN",
        "market_index": "VNINDEX / verify"
    },
    {
        "country": "Thailand",
        "company": "PTT",
        "ticker": "PTT.BK",
        "market_index": "^SET.BK"
    },
    {
        "country": "Thailand",
        "company": "PTT Exploration and Production",
        "ticker": "PTTEP.BK",
        "market_index": "^SET.BK"
    },
    {
        "country": "Thailand",
        "company": "CP All",
        "ticker": "CPALL.BK",
        "market_index": "^SET.BK"
    },
    {
        "country": "Thailand",
        "company": "Airports of Thailand",
        "ticker": "AOT.BK",
        "market_index": "^SET.BK"
    },
    {
        "country": "Thailand",
        "company": "Advanced Info Service",
        "ticker": "ADVANC.BK",
        "market_index": "^SET.BK"
    },
    {
        "country": "Thailand",
        "company": "Delta Electronics Thailand",
        "ticker": "DELTA.BK",
        "market_index": "^SET.BK"
    },
    {
        "country": "Thailand",
        "company": "Bangkok Dusit Medical Services",
        "ticker": "BDMS.BK",
        "market_index": "^SET.BK"
    },
    {
        "country": "Thailand",
        "company": "SCB X",
        "ticker": "SCB.BK",
        "market_index": "^SET.BK"
    },
    {
        "country": "Indonesia",
        "company": "Bank Central Asia",
        "ticker": "BBCA.JK",
        "market_index": "^JKSE"
    },
    {
        "country": "Indonesia",
        "company": "Bank Rakyat Indonesia",
        "ticker": "BBRI.JK",
        "market_index": "^JKSE"
    },
    {
        "country": "Indonesia",
        "company": "Bank Mandiri",
        "ticker": "BMRI.JK",
        "market_index": "^JKSE"
    },
    {
        "country": "Indonesia",
        "company": "Telkom Indonesia",
        "ticker": "TLKM.JK",
        "market_index": "^JKSE"
    },
    {
        "country": "Indonesia",
        "company": "Astra International",
        "ticker": "ASII.JK",
        "market_index": "^JKSE"
    },
    {
        "country": "Indonesia",
        "company": "Indofood CBP",
        "ticker": "ICBP.JK",
        "market_index": "^JKSE"
    },
    {
        "country": "Indonesia",
        "company": "Unilever Indonesia",
        "ticker": "UNVR.JK",
        "market_index": "^JKSE"
    },
    {
        "country": "Indonesia",
        "company": "Bank Negara Indonesia",
        "ticker": "BBNI.JK",
        "market_index": "^JKSE"
    },
    {
        "country": "India",
        "company": "Reliance Industries",
        "ticker": "RELIANCE.NS",
        "market_index": "^NSEI"
    },
    {
        "country": "India",
        "company": "Tata Consultancy Services",
        "ticker": "TCS.NS",
        "market_index": "^NSEI"
    },
    {
        "country": "India",
        "company": "HDFC Bank",
        "ticker": "HDFCBANK.NS",
        "market_index": "^NSEI"
    },
    {
        "country": "India",
        "company": "Infosys",
        "ticker": "INFY.NS",
        "market_index": "^NSEI"
    },
    {
        "country": "India",
        "company": "ICICI Bank",
        "ticker": "ICICIBANK.NS",
        "market_index": "^NSEI"
    },
    {
        "country": "India",
        "company": "Bharti Airtel",
        "ticker": "BHARTIARTL.NS",
        "market_index": "^NSEI"
    },
    {
        "country": "India",
        "company": "Sun Pharmaceutical",
        "ticker": "SUNPHARMA.NS",
        "market_index": "^NSEI"
    },
    {
        "country": "India",
        "company": "Tata Motors",
        "ticker": "TATAMOTORS.NS",
        "market_index": "^NSEI"
    },
    {
        "country": "South Korea",
        "company": "Samsung Electronics",
        "ticker": "005930.KS",
        "market_index": "^KS11"
    },
    {
        "country": "South Korea",
        "company": "SK Hynix",
        "ticker": "000660.KS",
        "market_index": "^KS11"
    },
    {
        "country": "South Korea",
        "company": "Hyundai Motor",
        "ticker": "005380.KS",
        "market_index": "^KS11"
    },
    {
        "country": "South Korea",
        "company": "Samsung Biologics",
        "ticker": "207940.KS",
        "market_index": "^KS11"
    },
    {
        "country": "South Korea",
        "company": "KB Financial Group",
        "ticker": "105560.KS",
        "market_index": "^KS11"
    },
    {
        "country": "South Korea",
        "company": "LG Energy Solution",
        "ticker": "373220.KS",
        "market_index": "^KS11"
    },
    {
        "country": "South Korea",
        "company": "POSCO Holdings",
        "ticker": "005490.KS",
        "market_index": "^KS11"
    },
    {
        "country": "South Korea",
        "company": "Naver",
        "ticker": "035420.KS",
        "market_index": "^KS11"
    },
    {
        "country": "Japan",
        "company": "Toyota Motor",
        "ticker": "7203.T",
        "market_index": "^N225"
    },
    {
        "country": "Japan",
        "company": "Sony Group",
        "ticker": "6758.T",
        "market_index": "^N225"
    },
    {
        "country": "Japan",
        "company": "Mitsubishi UFJ Financial",
        "ticker": "8306.T",
        "market_index": "^N225"
    },
    {
        "country": "Japan",
        "company": "Keyence",
        "ticker": "6861.T",
        "market_index": "^N225"
    },
    {
        "country": "Japan",
        "company": "Hitachi",
        "ticker": "6501.T",
        "market_index": "^N225"
    },
    {
        "country": "Japan",
        "company": "Tokyo Electron",
        "ticker": "8035.T",
        "market_index": "^N225"
    },
    {
        "country": "Japan",
        "company": "Nintendo",
        "ticker": "7974.T",
        "market_index": "^N225"
    },
    {
        "country": "Japan",
        "company": "Recruit Holdings",
        "ticker": "6098.T",
        "market_index": "^N225"
    },
    {
        "country": "Malaysia",
        "company": "Maybank",
        "ticker": "1155.KL",
        "market_index": "^KLSE"
    },
    {
        "country": "Malaysia",
        "company": "CIMB Group",
        "ticker": "1023.KL",
        "market_index": "^KLSE"
    },
    {
        "country": "Malaysia",
        "company": "Public Bank",
        "ticker": "1295.KL",
        "market_index": "^KLSE"
    },
    {
        "country": "Malaysia",
        "company": "Tenaga Nasional",
        "ticker": "5347.KL",
        "market_index": "^KLSE"
    },
    {
        "country": "Malaysia",
        "company": "Petronas Gas",
        "ticker": "6033.KL",
        "market_index": "^KLSE"
    },
    {
        "country": "Malaysia",
        "company": "IHH Healthcare",
        "ticker": "5225.KL",
        "market_index": "^KLSE"
    },
    {
        "country": "Malaysia",
        "company": "Inari Amertron",
        "ticker": "0166.KL",
        "market_index": "^KLSE"
    },
    {
        "country": "Malaysia",
        "company": "Top Glove",
        "ticker": "7113.KL",
        "market_index": "^KLSE"
    },
    {
        "country": "Switzerland",
        "company": "Nestlé",
        "ticker": "NESN.SW",
        "market_index": "^SSMI"
    },
    {
        "country": "Switzerland",
        "company": "Roche Holding",
        "ticker": "ROG.SW",
        "market_index": "^SSMI"
    },
    {
        "country": "Switzerland",
        "company": "Novartis",
        "ticker": "NOVN.SW",
        "market_index": "^SSMI"
    },
    {
        "country": "Switzerland",
        "company": "UBS Group",
        "ticker": "UBSG.SW",
        "market_index": "^SSMI"
    },
    {
        "country": "Switzerland",
        "company": "ABB",
        "ticker": "ABBN.SW",
        "market_index": "^SSMI"
    },
    {
        "country": "Switzerland",
        "company": "Zurich Insurance",
        "ticker": "ZURN.SW",
        "market_index": "^SSMI"
    },
    {
        "country": "Switzerland",
        "company": "Richemont",
        "ticker": "CFR.SW",
        "market_index": "^SSMI"
    },
    {
        "country": "Germany",
        "company": "SAP",
        "ticker": "SAP.DE",
        "market_index": "^GDAXI"
    },
    {
        "country": "Germany",
        "company": "Siemens",
        "ticker": "SIE.DE",
        "market_index": "^GDAXI"
    },
    {
        "country": "Germany",
        "company": "Allianz",
        "ticker": "ALV.DE",
        "market_index": "^GDAXI"
    },
    {
        "country": "Germany",
        "company": "Deutsche Telekom",
        "ticker": "DTE.DE",
        "market_index": "^GDAXI"
    },
    {
        "country": "Germany",
        "company": "Mercedes-Benz Group",
        "ticker": "MBG.DE",
        "market_index": "^GDAXI"
    },
    {
        "country": "Germany",
        "company": "BMW",
        "ticker": "BMW.DE",
        "market_index": "^GDAXI"
    },
    {
        "country": "Germany",
        "company": "BASF",
        "ticker": "BAS.DE",
        "market_index": "^GDAXI"
    },
    {
        "country": "France",
        "company": "LVMH",
        "ticker": "MC.PA",
        "market_index": "^FCHI"
    },
    {
        "country": "France",
        "company": "TotalEnergies",
        "ticker": "TTE.PA",
        "market_index": "^FCHI"
    },
    {
        "country": "France",
        "company": "L'Oréal",
        "ticker": "OR.PA",
        "market_index": "^FCHI"
    },
    {
        "country": "France",
        "company": "Sanofi",
        "ticker": "SAN.PA",
        "market_index": "^FCHI"
    },
    {
        "country": "France",
        "company": "Schneider Electric",
        "ticker": "SU.PA",
        "market_index": "^FCHI"
    },
    {
        "country": "France",
        "company": "BNP Paribas",
        "ticker": "BNP.PA",
        "market_index": "^FCHI"
    },
    {
        "country": "France",
        "company": "Airbus",
        "ticker": "AIR.PA",
        "market_index": "^FCHI"
    },
    {
        "country": "United Kingdom",
        "company": "AstraZeneca",
        "ticker": "AZN.L",
        "market_index": "^FTSE"
    },
    {
        "country": "United Kingdom",
        "company": "Shell",
        "ticker": "SHEL.L",
        "market_index": "^FTSE"
    },
    {
        "country": "United Kingdom",
        "company": "HSBC Holdings",
        "ticker": "HSBA.L",
        "market_index": "^FTSE"
    },
    {
        "country": "United Kingdom",
        "company": "Unilever",
        "ticker": "ULVR.L",
        "market_index": "^FTSE"
    },
    {
        "country": "United Kingdom",
        "company": "BP",
        "ticker": "BP.L",
        "market_index": "^FTSE"
    },
    {
        "country": "United Kingdom",
        "company": "Rio Tinto",
        "ticker": "RIO.L",
        "market_index": "^FTSE"
    },
    {
        "country": "United Kingdom",
        "company": "Diageo",
        "ticker": "DGE.L",
        "market_index": "^FTSE"
    },
    {
        "country": "United Kingdom",
        "company": "BAE Systems",
        "ticker": "BA.L",
        "market_index": "^FTSE"
    },
    {
        "country": "Australia",
        "company": "BHP Group",
        "ticker": "BHP.AX",
        "market_index": "^AXJO"
    },
    {
        "country": "Australia",
        "company": "Commonwealth Bank",
        "ticker": "CBA.AX",
        "market_index": "^AXJO"
    },
    {
        "country": "Australia",
        "company": "CSL Limited",
        "ticker": "CSL.AX",
        "market_index": "^AXJO"
    },
    {
        "country": "Australia",
        "company": "Woodside Energy",
        "ticker": "WDS.AX",
        "market_index": "^AXJO"
    },
    {
        "country": "Australia",
        "company": "Wesfarmers",
        "ticker": "WES.AX",
        "market_index": "^AXJO"
    },
    {
        "country": "Australia",
        "company": "National Australia Bank",
        "ticker": "NAB.AX",
        "market_index": "^AXJO"
    },
    {
        "country": "Australia",
        "company": "Macquarie Group",
        "ticker": "MQG.AX",
        "market_index": "^AXJO"
    },
    {
        "country": "Australia",
        "company": "Fortescue",
        "ticker": "FMG.AX",
        "market_index": "^AXJO"
    },
    {
        "country": "Brazil",
        "company": "Petrobras",
        "ticker": "PETR4.SA",
        "market_index": "^BVSP"
    },
    {
        "country": "Brazil",
        "company": "Vale",
        "ticker": "VALE3.SA",
        "market_index": "^BVSP"
    },
    {
        "country": "Brazil",
        "company": "Itaú Unibanco",
        "ticker": "ITUB4.SA",
        "market_index": "^BVSP"
    },
    {
        "country": "Brazil",
        "company": "Ambev",
        "ticker": "ABEV3.SA",
        "market_index": "^BVSP"
    },
    {
        "country": "Brazil",
        "company": "WEG",
        "ticker": "WEGE3.SA",
        "market_index": "^BVSP"
    },
    {
        "country": "Brazil",
        "company": "Banco do Brasil",
        "ticker": "BBAS3.SA",
        "market_index": "^BVSP"
    },
    {
        "country": "Brazil",
        "company": "JBS",
        "ticker": "JBSS3.SA",
        "market_index": "^BVSP"
    },
    {
        "country": "Singapore",
        "company": "DBS Group Holdings",
        "ticker": "D05.SI",
        "market_index": "^STI"
    },
    {
        "country": "Singapore",
        "company": "Sea Limited",
        "ticker": "SE",
        "market_index": "^STI"
    },
    {
        "country": "Singapore",
        "company": "Singapore Telecom",
        "ticker": "Z74.SI",
        "market_index": "^STI"
    },
    {
        "country": "Singapore",
        "company": "Wilmar International",
        "ticker": "F34.SI",
        "market_index": "^STI"
    },
    {
        "country": "Singapore",
        "company": "OCBC Bank",
        "ticker": "O39.SI",
        "market_index": "^STI"
    },
    {
        "country": "Singapore",
        "company": "UOB",
        "ticker": "U11.SI",
        "market_index": "^STI"
    },
    {
        "country": "Singapore",
        "company": "CapitaLand Investment",
        "ticker": "9CI.SI",
        "market_index": "^STI"
    },
    {
        "country": "Italy",
        "company": "Eni",
        "ticker": "ENI.MI",
        "market_index": "FTSEMIB.MI"
    },
    {
        "country": "Italy",
        "company": "UniCredit",
        "ticker": "UCG.MI",
        "market_index": "FTSEMIB.MI"
    },
    {
        "country": "Italy",
        "company": "Intesa Sanpaolo",
        "ticker": "ISP.MI",
        "market_index": "FTSEMIB.MI"
    },
    {
        "country": "Italy",
        "company": "Ferrari",
        "ticker": "RACE.MI",
        "market_index": "FTSEMIB.MI"
    },
    {
        "country": "Italy",
        "company": "Enel",
        "ticker": "ENEL.MI",
        "market_index": "FTSEMIB.MI"
    },
    {
        "country": "Italy",
        "company": "Stellantis",
        "ticker": "STLAM.MI",
        "market_index": "FTSEMIB.MI"
    },
    {
        "country": "Italy",
        "company": "Leonardo",
        "ticker": "LDO.MI",
        "market_index": "FTSEMIB.MI"
    },
    {
        "country": "Spain",
        "company": "Banco Santander",
        "ticker": "SAN.MC",
        "market_index": "^IBEX"
    },
    {
        "country": "Spain",
        "company": "Iberdrola",
        "ticker": "IBE.MC",
        "market_index": "^IBEX"
    },
    {
        "country": "Spain",
        "company": "Inditex",
        "ticker": "ITX.MC",
        "market_index": "^IBEX"
    },
    {
        "country": "Spain",
        "company": "BBVA",
        "ticker": "BBVA.MC",
        "market_index": "^IBEX"
    },
    {
        "country": "Spain",
        "company": "Telefónica",
        "ticker": "TEF.MC",
        "market_index": "^IBEX"
    },
    {
        "country": "Spain",
        "company": "Repsol",
        "ticker": "REP.MC",
        "market_index": "^IBEX"
    },
    {
        "country": "Spain",
        "company": "Amadeus IT Group",
        "ticker": "AMS.MC",
        "market_index": "^IBEX"
    },
    {
        "country": "Netherlands",
        "company": "ASML",
        "ticker": "ASML.AS",
        "market_index": "^AEX"
    },
    {
        "country": "Netherlands",
        "company": "Prosus",
        "ticker": "PRX.AS",
        "market_index": "^AEX"
    },
    {
        "country": "Netherlands",
        "company": "Adyen",
        "ticker": "ADYEN.AS",
        "market_index": "^AEX"
    },
    {
        "country": "Netherlands",
        "company": "ING Group",
        "ticker": "INGA.AS",
        "market_index": "^AEX"
    },
    {
        "country": "Netherlands",
        "company": "Philips",
        "ticker": "PHIA.AS",
        "market_index": "^AEX"
    },
    {
        "country": "Netherlands",
        "company": "Heineken",
        "ticker": "HEIA.AS",
        "market_index": "^AEX"
    },
    {
        "country": "Netherlands",
        "company": "Ahold Delhaize",
        "ticker": "AD.AS",
        "market_index": "^AEX"
    },
    {
        "country": "Netherlands",
        "company": "DSM-Firmenich",
        "ticker": "DSFIR.AS",
        "market_index": "^AEX"
    },
    {
        "country": "South Africa",
        "company": "Naspers",
        "ticker": "NPN.JO",
        "market_index": "J203.JO / verify"
    },
    {
        "country": "South Africa",
        "company": "Anglo American",
        "ticker": "AGL.JO",
        "market_index": "J203.JO / verify"
    },
    {
        "country": "South Africa",
        "company": "Sasol",
        "ticker": "SOL.JO",
        "market_index": "J203.JO / verify"
    },
    {
        "country": "South Africa",
        "company": "MTN Group",
        "ticker": "MTN.JO",
        "market_index": "J203.JO / verify"
    },
    {
        "country": "South Africa",
        "company": "Standard Bank Group",
        "ticker": "SBK.JO",
        "market_index": "J203.JO / verify"
    },
    {
        "country": "South Africa",
        "company": "FirstRand",
        "ticker": "FSR.JO",
        "market_index": "J203.JO / verify"
    },
    {
        "country": "South Africa",
        "company": "Shoprite Holdings",
        "ticker": "SHP.JO",
        "market_index": "J203.JO / verify"
    }
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================


def clean_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def clean_market_index(country, ticker, market_index):
    """Clean market index entries and apply country fallback if needed."""
    country = clean_text(country)
    ticker = clean_text(ticker)
    mi = clean_text(market_index)

    # Handle entries like "VNINDEX / verify" or "J203.JO / verify".
    if "/" in mi:
        mi = mi.split("/")[0].strip()

    if "verify" in mi.lower() or mi == "" or mi.lower() == "nan":
        mi = DEFAULT_MARKET_INDEX.get(country, mi)

    # Specific fallbacks for known ambiguous entries.
    if country == "Vietnam" and (mi.upper() == "VNINDEX" or mi == ""):
        mi = DEFAULT_MARKET_INDEX["Vietnam"]
    if country == "South Africa" and (mi.upper() == "J203.JO" or mi == ""):
        mi = DEFAULT_MARKET_INDEX["South Africa"]

    # China: Hong Kong-listed firms should use HSI, A-share firms CSI 300.
    if country == "China":
        if ticker.endswith(".HK"):
            mi = "^HSI"
        elif ticker.endswith(".SS") or ticker.endswith(".SZ"):
            mi = "000300.SS"

    return mi


def build_firm_dataframe():
    """Convert embedded FIRM_LIST to a clean dataframe."""
    df = pd.DataFrame(FIRM_LIST)
    required = ["country", "company", "ticker", "market_index"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"FIRM_LIST is missing columns: {missing}")

    df = df.rename(columns={
        "country": "Country",
        "company": "Company",
        "ticker": "Ticker",
        "market_index": "Market_Index",
    })
    for col in ["Country", "Company", "Ticker", "Market_Index"]:
        df[col] = df[col].apply(clean_text)

    df["Market_Index"] = df.apply(
        lambda r: clean_market_index(r["Country"], r["Ticker"], r["Market_Index"]),
        axis=1,
    )

    # Remove empty tickers and duplicated company rows.
    df = df[(df["Ticker"] != "") & (df["Market_Index"] != "")]
    df = df.drop_duplicates(subset=["Country", "Company", "Ticker"]).reset_index(drop=True)
    return df


def extract_close_prices(downloaded_data):
    """Extract close prices from yfinance output robustly."""
    if downloaded_data is None or downloaded_data.empty:
        return pd.DataFrame()

    if isinstance(downloaded_data.columns, pd.MultiIndex):
        if "Close" in downloaded_data.columns.get_level_values(0):
            close = downloaded_data["Close"].copy()
        elif "Adj Close" in downloaded_data.columns.get_level_values(0):
            close = downloaded_data["Adj Close"].copy()
        else:
            raise ValueError("Could not find Close or Adj Close in yfinance output.")
    else:
        if "Close" in downloaded_data.columns:
            close = downloaded_data[["Close"]].copy()
        elif "Adj Close" in downloaded_data.columns:
            close = downloaded_data[["Adj Close"]].copy()
        else:
            raise ValueError("Could not find Close or Adj Close in yfinance output.")
    return close


def download_prices(tickers, start, end, batch_size=40):
    """Download prices in batches to reduce timeout risk."""
    tickers = [t for t in pd.unique(pd.Series(tickers).dropna()) if str(t).strip()]
    all_prices = []

    print(f"\n[1/4] Downloading adjusted close prices from Yahoo Finance")
    print(f"      Period: {start} to {end}")
    print(f"      Unique tickers/indices: {len(tickers)}")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"      Batch {i // batch_size + 1}/{(len(tickers) - 1) // batch_size + 1}: {len(batch)} symbols")
        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            close = extract_close_prices(data)
            if len(batch) == 1 and close.shape[1] == 1:
                close.columns = batch
            all_prices.append(close)
        except Exception as e:
            print(f"      Warning: batch failed: {e}")

    if not all_prices:
        raise RuntimeError("No price data downloaded. Check internet connection and tickers.")

    prices = pd.concat(all_prices, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()
    print(f"      Downloaded price matrix: {prices.shape[0]} days × {prices.shape[1]} symbols")
    return prices


def choose_local_event_day(pair_df, event_date):
    """Choose event trading day for a firm-index pair.

    Prefer the exact event date. If unavailable, use the next available trading day.
    If no later day exists, use the closest available trading day.
    """
    event_dt = pd.Timestamp(event_date)
    idx = pair_df.index

    if event_dt in idx:
        return event_dt, "exact"

    later = idx[idx >= event_dt]
    if len(later) > 0:
        return later[0], "next_available"

    diffs = np.abs((idx - event_dt).days)
    pos = int(np.argmin(diffs))
    return idx[pos], "closest_available"


def estimate_market_model_and_car(row, returns, event_date, min_est_days):
    """Estimate market model and calculate AR/CAR for one firm."""
    ticker = row["Ticker"]
    market_index = row["Market_Index"]

    if ticker not in returns.columns:
        raise ValueError(f"No firm return data downloaded for {ticker}")
    if market_index not in returns.columns:
        raise ValueError(f"No market index return data downloaded for {market_index}")

    pair = pd.DataFrame({
        "firm": returns[ticker],
        "market": returns[market_index],
    }).dropna()

    if pair.empty:
        raise ValueError("No overlapping firm-market return observations")

    event_day, event_day_rule = choose_local_event_day(pair, event_date)
    firm_days = pair.index.tolist()
    t0 = firm_days.index(event_day)

    est_start = t0 + EST_WINDOW_START
    est_end = t0 + EST_WINDOW_END
    evt_start = t0 + EVT_WINDOW_START
    evt_end = t0 + EVT_WINDOW_END

    if est_start < 0:
        raise ValueError(f"Estimation window starts before available data. Need more pre-event data; available start index {t0}")
    if evt_start < 0 or evt_end >= len(firm_days):
        raise ValueError("Event window out of available data range")

    est_data = pair.iloc[est_start:est_end + 1].dropna()
    if len(est_data) < min_est_days:
        raise ValueError(f"Insufficient estimation-window observations: {len(est_data)} < {min_est_days}")

    # Market model: R_i,t = alpha_i + beta_i * R_m,t + eps_i,t
    X = sm.add_constant(est_data["market"])
    y = est_data["firm"]
    model = sm.OLS(y, X).fit()
    alpha = model.params["const"]
    beta = model.params["market"]
    r2 = model.rsquared

    evt_data = pair.iloc[evt_start:evt_end + 1].copy()
    evt_data["expected"] = alpha + beta * evt_data["market"]
    evt_data["AR"] = evt_data["firm"] - evt_data["expected"]

    # Event window should contain exactly 3 trading observations for [-1,+1].
    if len(evt_data) != 3:
        raise ValueError(f"Event window has {len(evt_data)} observations, expected 3")

    ar_minus1 = evt_data["AR"].iloc[0]
    ar_0 = evt_data["AR"].iloc[1]
    ar_plus1 = evt_data["AR"].iloc[2]
    car = evt_data["AR"].sum()

    return {
        "Country": row["Country"],
        "Company": row["Company"],
        "Ticker": ticker,
        "Market_Index": market_index,
        "Event_Day_Used": event_day.strftime("%Y-%m-%d"),
        "Event_Day_Rule": event_day_rule,
        "Alpha": alpha,
        "Beta": beta,
        "R_squared": r2,
        "N_estimation_days": len(est_data),
        "AR[-1]": ar_minus1,
        "AR[0]": ar_0,
        "AR[+1]": ar_plus1,
        "CAR[-1,+1]": car,
        "CAR_pct": car * 100,
    }


def print_results(results_df, errors_df):
    """Print terminal-friendly results and summary."""
    print("\n[4/4] Results")
    if results_df.empty:
        print("      No CARs were successfully calculated.")
        return

    display_cols = ["Country", "Company", "Ticker", "Market_Index", "CAR_pct", "Alpha", "Beta", "R_squared", "N_estimation_days"]
    print("\n=== Firm-level CAR results, sorted by CAR ===")
    print(
        results_df.sort_values("CAR_pct")[display_cols]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    country_summary = results_df.groupby("Country").agg(
        N=("CAR_pct", "count"),
        Mean_CAR_pct=("CAR_pct", "mean"),
        Median_CAR_pct=("CAR_pct", "median"),
        Std_CAR_pct=("CAR_pct", "std"),
        Negative_pct=("CAR_pct", lambda s: (s < 0).mean() * 100),
    ).sort_values("Mean_CAR_pct")

    print("\n=== Country summary ===")
    print(country_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    mean_car = results_df["CAR_pct"].mean()
    std_car = results_df["CAR_pct"].std()
    n = len(results_df)
    t_stat = mean_car / (std_car / np.sqrt(n)) if n > 1 and std_car > 0 else np.nan
    p_val = stats.t.sf(abs(t_stat), n - 1) * 2 if np.isfinite(t_stat) and n > 1 else np.nan

    print("\n=== Full-sample CAR test ===")
    print(f"N                         : {n}")
    print(f"Mean CAR[-1,+1] (%)       : {mean_car:.4f}")
    print(f"Median CAR[-1,+1] (%)     : {results_df['CAR_pct'].median():.4f}")
    print(f"Std. Dev. CAR (%)         : {std_car:.4f}")
    print(f"% Negative CAR            : {(results_df['CAR_pct'] < 0).mean() * 100:.2f}%")
    print(f"t-statistic H0: mean=0    : {t_stat:.4f}")
    print(f"two-sided p-value         : {p_val:.4f}")

    if not errors_df.empty:
        print("\n=== Errors / skipped firms ===")
        print(errors_df.to_string(index=False))


def save_csv_outputs(results_df, errors_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not results_df.empty:
        results_path = os.path.join(output_dir, "CAR_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved firm-level results to: {results_path}")

        country_path = os.path.join(output_dir, "Country_summary.csv")
        results_df.groupby("Country").agg(
            N=("CAR_pct", "count"),
            Mean_CAR_pct=("CAR_pct", "mean"),
            Median_CAR_pct=("CAR_pct", "median"),
            Std_CAR_pct=("CAR_pct", "std"),
            Negative_pct=("CAR_pct", lambda s: (s < 0).mean() * 100),
        ).to_csv(country_path)
        print(f"Saved country summary to: {country_path}")

    if not errors_df.empty:
        errors_path = os.path.join(output_dir, "Errors.csv")
        errors_df.to_csv(errors_path, index=False)
        print(f"Saved errors to: {errors_path}")


# ============================================================
# 4. MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Calculate CAR[-1,+1] using embedded Yahoo Finance tickers.")
    parser.add_argument("--event-date", default=EVENT_DATE, help="Event date in YYYY-MM-DD format. Default: 2025-04-02")
    parser.add_argument("--start", default=DOWNLOAD_START, help="Download start date. Default: 2023-06-01")
    parser.add_argument("--end", default=DOWNLOAD_END, help="Download end date. Default: 2025-04-11")
    parser.add_argument("--min-est-days", type=int, default=MIN_EST_DAYS, help="Minimum estimation-window observations. Default: 192")
    parser.add_argument("--save-csv", action="store_true", help="Save CSV outputs to output directory. No Excel output is ever created.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="CSV output directory if --save-csv is used.")
    args = parser.parse_args()

    print("=" * 72)
    print("  Event Study CAR Calculation — Embedded Ticker Version")
    print("=" * 72)
    print(f"Event date          : {args.event_date}")
    print(f"Estimation window   : [{EST_WINDOW_START}, {EST_WINDOW_END}]")
    print(f"Event window        : [{EVT_WINDOW_START}, +{EVT_WINDOW_END}]")
    print(f"Minimum est. days   : {args.min_est_days}")
    print("Excel output        : Disabled")

    firm_df = build_firm_dataframe()
    print(f"\nEmbedded company list: {len(firm_df)} firms across {firm_df['Country'].nunique()} countries")

    print("\nCountry counts:")
    print(firm_df["Country"].value_counts().sort_index().to_string())

    # Build unique ticker/index list.
    all_symbols = list(pd.unique(pd.concat([firm_df["Ticker"], firm_df["Market_Index"]], ignore_index=True)))

    prices = download_prices(all_symbols, args.start, args.end)

    print("\n[2/4] Computing daily log returns from adjusted close prices")
    returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    print(f"      Return matrix: {returns.shape[0]} days × {returns.shape[1]} symbols")

    print("\n[3/4] Estimating market models and calculating AR/CAR")
    results = []
    errors = []

    for idx, row in firm_df.iterrows():
        try:
            res = estimate_market_model_and_car(row, returns, args.event_date, args.min_est_days)
            results.append(res)
            print(f"  [{idx + 1:>3}/{len(firm_df)}] OK   {row['Ticker']:<12} {row['Company'][:36]:<36} CAR = {res['CAR_pct']:>8.3f}%")
        except Exception as e:
            errors.append({
                "Country": row["Country"],
                "Company": row["Company"],
                "Ticker": row["Ticker"],
                "Market_Index": row["Market_Index"],
                "Error": str(e),
            })
            print(f"  [{idx + 1:>3}/{len(firm_df)}] FAIL {row['Ticker']:<12} {row['Company'][:36]:<36} {e}")

    results_df = pd.DataFrame(results)
    errors_df = pd.DataFrame(errors)

    print(f"\nSuccessfully calculated CARs: {len(results_df)}")
    print(f"Errors/skipped firms        : {len(errors_df)}")

    print_results(results_df, errors_df)

    if args.save_csv:
        save_csv_outputs(results_df, errors_df, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
