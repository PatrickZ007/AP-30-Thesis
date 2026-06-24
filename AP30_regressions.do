*==============================================================================*
* AP-30  Group 3 Thesis  --  Table 7  Cross-Sectional Regression Results
*
* "Liberation Day" tariff shock (2 April 2025).
* Dependent variable : CAR[-1,+1] in PERCENT  (CAR * 100)
* Main variable       : Excess Tariff = (tariff in percentage points) demeaned
*                       i.e. tariff_pp - mean(tariff_pp) over the 114-firm sample
* Moderator           : FDc  (financial development, credit/GDP)
* Interaction         : FDc x Excess Tariff                         -> H2
* Firm controls       : ln(Total assets), Leverage, Book-to-market
* Standard errors     : clustered by country (16 clusters)
*
* Four nested models (the four columns of Table 7):
*   (1) Excess Tariff only
*   (2) Excess Tariff + firm controls
*   (3) Excess Tariff + FDc + FDc x Excess Tariff
*   (4) Full  (Excess Tariff + FDc + FDc x Excess Tariff + firm controls)
*
* INPUT : dependent_variables.xlsx   (sheet "Sheet1", 114 firms / 16 countries)
*         columns: Country  Company  tariff  FDc  CAR  Total_Assets_USD
*                  Leverage  BM_Ratio
*
* HOW TO RUN
*   1. Put dependent_variables.xlsx in the folder named in "global root" below.
*   2. Open this file in Stata (File > Open), then press the Do button (the >).
*   That's it - you do not type anything anywhere else.
*==============================================================================*

clear all
set more off
version 14

* ---- set this to the folder that holds dependent_variables.xlsx --------------*
* (Change the path if your file lives somewhere other than Downloads.)
global root "/Users/hasanbleda/Downloads"
cd "$root"

cap log close
log using "AP30_table7_output.log", replace text   // raw output -> Appendix

*------------------------------------------------------------------------------*
* 1. Import the data
*------------------------------------------------------------------------------*
import excel using "dependent_variables.xlsx", sheet("Sheet1") firstrow clear

* keep only the variables we need and drop any blank trailing rows
keep Country Company tariff FDc CAR Total_Assets_USD Leverage BM_Ratio
drop if missing(CAR)

*------------------------------------------------------------------------------*
* 2. Build the regression variables
*------------------------------------------------------------------------------*
* Dependent variable in percentage points
gen CAR_pct = CAR * 100

* Tariff in percentage points, then demeaned -> "Excess Tariff"
gen tariff_pp = tariff * 100
quietly summarize tariff_pp
gen ExcessTariff = tariff_pp - r(mean)

* Moderator and interaction (H2)
gen FDc_x_Tariff = FDc * ExcessTariff

* Firm controls
gen lnAssets = ln(Total_Assets_USD)
* Leverage and BM_Ratio are used as-is

* Country identifier for clustering
encode Country, gen(country_id)

label var ExcessTariff  "Excess Tariff"
label var FDc           "FDc"
label var FDc_x_Tariff  "FDc x Tariff shock"
label var lnAssets      "ln(Total assets)"
label var Leverage      "Leverage"
label var BM_Ratio      "Book-to-market"

*------------------------------------------------------------------------------*
* 3. The four nested regressions (country-clustered SEs)
*------------------------------------------------------------------------------*
eststo clear

* (1) Excess Tariff only
eststo M1: regress CAR_pct ExcessTariff, vce(cluster country_id)

* (2) Excess Tariff + firm controls
eststo M2: regress CAR_pct ExcessTariff lnAssets Leverage BM_Ratio, ///
        vce(cluster country_id)

* (3) Excess Tariff + FDc + FDc x Excess Tariff
eststo M3: regress CAR_pct ExcessTariff FDc FDc_x_Tariff, vce(cluster country_id)

* (4) Full model
eststo M4: regress CAR_pct ExcessTariff FDc FDc_x_Tariff lnAssets Leverage BM_Ratio, ///
        vce(cluster country_id)

*------------------------------------------------------------------------------*
* 4. Combined side-by-side table (Table 7)
*    Needs the free add-on estout.  If you do not have it, run once:
*         ssc install estout
*    The four regressions above still print on their own without it.
*------------------------------------------------------------------------------*
cap which esttab
if _rc==0 {
    esttab M1 M2 M3 M4, ///
        b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
        scalars("N Observations" "N_clust Country clusters" ///
                "r2 R-squared" "r2_a Adjusted R-squared" "F F-statistic") ///
        order(ExcessTariff FDc FDc_x_Tariff lnAssets Leverage BM_Ratio _cons) ///
        mtitles("(1)" "(2)" "(3)" "(4)") ///
        title("Table 7. Cross-Sectional Regression Results") ///
        nonumbers compress

    * Word-friendly copy you can open in Word:
    esttab M1 M2 M3 M4 using "AP30_Table7.rtf", replace ///
        b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
        scalars("N Observations" "N_clust Country clusters" ///
                "r2 R-squared" "r2_a Adjusted R-squared" "F F-statistic") ///
        order(ExcessTariff FDc FDc_x_Tariff lnAssets Leverage BM_Ratio _cons) ///
        mtitles("(1)" "(2)" "(3)" "(4)") nonumbers ///
        title("Table 7. Cross-Sectional Regression Results")
}
else {
    di as txt "estout not installed - the four model blocks above are your results."
    di as txt "To get the combined table, type:  ssc install estout   then re-run."
}

log close

*------------------------------------------------------------------------------*
* Expected numbers (so you can confirm the run matches Table 7):
*   Excess Tariff   (1) 0.0170  (2) 0.0026  (3) 0.1042  (4) 0.0719
*   FDc x Tariff    (3) -0.0579 (4) -0.0454
*   Constant        (1) -0.6696**  (3) -0.6003
*   N = 114, Country clusters = 16
*   R-squared       (1) 0.0018 (2) 0.0240 (3) 0.0077 (4) 0.0270
*   F-statistic     (1) 0.2795 (2) 0.6313 (3) 0.7934 (4) 0.8273
*==============================================================================*
