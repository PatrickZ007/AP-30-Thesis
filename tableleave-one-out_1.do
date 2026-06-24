*===============================================================
*  TABLE 10 - Country-level influence on the FDc x Tariff
*  interaction (leave-one-country-out jackknife)
*
*  Reproduces the baseline interaction (-0.0454) and the
*  per-country deviations and p-values reported in the thesis.
*
*  Data: dependent_variables.xlsx  (16 countries, 114 firms)
*  Model: full specification of Table 7
*         CAR  =  a + b1*tariff + b2*FDc + b3*(tariff x FDc)
*                   + size + leverage + book-to-market + e
*         standard errors clustered by country.
*
*  Requires the boottest package for the wild cluster bootstrap
*  step (Section 2b):   ssc install boottest
*===============================================================
version 14
clear all
set more off

*---------------------------------------------------------------
* 0.  Import data  (edit the path to point at your file)
*---------------------------------------------------------------
import excel using "dependent_variables.xlsx", firstrow clear

* The sheet has a leading blank column and some trailing blank
* columns; they import as empty variables and are harmless.
* Run -describe- once to confirm the variable names below exist:
*   Country  tariff  FDc  CAR  Total_Assets_USD  Leverage  BM_Ratio
describe

*---------------------------------------------------------------
* 1.  Build variables
*---------------------------------------------------------------
* make sure the numeric fields are numeric (only destrings if needed)
foreach v in tariff FDc CAR Total_Assets_USD Leverage BM_Ratio {
    capture confirm numeric variable `v'
    if _rc destring `v', replace force
}

gen size = ln(Total_Assets_USD)          // firm size = ln(total assets)
rename Leverage leverage
rename BM_Ratio bm

* keep complete cases (matches the 114-firm estimation sample)
drop if missing(CAR, tariff, FDc, size, leverage, bm)
encode Country, gen(cid)                  // numeric cluster id (countries)

* NOTE on CAR units: CAR enters as stored in the file (decimal
* form), which reproduces the interaction coefficient of -0.0454.
* If you want CAR in percent instead, uncomment the next line
* (every coefficient then scales by 100):
* replace CAR = 100*CAR

*---------------------------------------------------------------
* 2.  Baseline interaction (all 16 countries)
*     Centering tariff/FDc would change only the lower-order
*     terms, not the interaction, so factor notation is used
*     directly (c.tariff##c.FDc).
*---------------------------------------------------------------
regress CAR c.tariff##c.FDc size leverage bm, vce(cluster cid)
scalar base = _b[c.tariff#c.FDc]
display as text "Baseline FDc x Tariff = " as result %7.4f base

*---------------------------------------------------------------
* 2b. Wild cluster bootstrap on the baseline interaction
*     (few-cluster-robust p-value; reproduces the 0.667).
*     Requires boottest:  ssc install boottest
*     Restricted (imposes the null) + Rademacher weights are the
*     defaults, the variant recommended for few clusters.
*---------------------------------------------------------------
set seed 12345                            // reproducibility
boottest c.tariff#c.FDc, reps(99999) weighttype(rademacher) nograph
scalar p_boot = r(p)
display as text "Wild cluster bootstrap p (interaction) = " as result %5.3f p_boot

*---------------------------------------------------------------
* 3.  Leave-one-country-out loop
*---------------------------------------------------------------
tempname P
postfile `P' str20 excluded double(fd tariff_pct coef dev p) ///
        using "table10_results.dta", replace

* --- baseline row (no country excluded) ---
quietly regress CAR c.tariff##c.FDc size leverage bm, vce(cluster cid)
scalar c0 = _b[c.tariff#c.FDc]
scalar t0 = _b[c.tariff#c.FDc]/_se[c.tariff#c.FDc]
scalar p0 = 2*(1-normal(abs(t0)))         // normal-based two-sided p (= 0.528)
post `P' ("None (baseline)") (.) (.) (c0) (0) (p0)

* --- one row per excluded country ---
levelsof Country, local(countries)
foreach c of local countries {
    quietly summarize FDc    if Country=="`c'", meanonly
    local fdv  = r(mean)
    quietly summarize tariff if Country=="`c'", meanonly
    local tarv = 100*r(mean)              // tariff displayed in percent

    quietly regress CAR c.tariff##c.FDc size leverage bm ///
            if Country!="`c'", vce(cluster cid)
    scalar cc = _b[c.tariff#c.FDc]
    scalar tt = _b[c.tariff#c.FDc]/_se[c.tariff#c.FDc]
    scalar pp = 2*(1-normal(abs(tt)))     // normal-based two-sided p

    *  To use Stata's native clustered p (t with G-1 df) instead,
    *  replace the line above with:
    *  scalar pp = 2*ttail(e(df_r), abs(tt))

    post `P' ("`c'") (`fdv') (`tarv') (cc) (cc-base) (pp)
}
postclose `P'

*---------------------------------------------------------------
* 4.  Assemble, order by absolute influence, add stars
*---------------------------------------------------------------
use "table10_results.dta", clear
gen absdev   = abs(dev)
gen byte top = (excluded=="None (baseline)")
gsort -top -absdev                        // baseline first, then |dev| desc
drop top absdev

gen str3 stars = ""
replace stars = "*"   if p<0.10
replace stars = "**"  if p<0.05
replace stars = "***" if p<0.01

format fd        %5.3f
format tariff_pct %4.0f
format coef dev  %8.4f
format p         %5.3f

label var excluded   "Country excluded"
label var fd         "FDc"
label var tariff_pct "Tariff (%)"
label var coef       "FDc x Tariff"
label var dev        "D vs baseline"
label var p          "p (interact.)"

list excluded fd tariff_pct coef dev p stars, ///
     noobs sep(0) abbrev(16)

*---------------------------------------------------------------
* 5.  (optional) export to CSV for pasting into the thesis
*---------------------------------------------------------------
* export delimited excluded fd tariff_pct coef dev p stars ///
*     using "table10.csv", replace

*===============================================================
* End of file
*===============================================================
