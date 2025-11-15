# About Dataset

### Context:
This dataset contains the medical records of 5000 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

### Attribute Information:
    - age: age of the patient (years)
    - anaemia: decrease of red blood cells or hemoglobin (boolean)
    - creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
    - diabetes: if the patient has diabetes (boolean)
    - ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
    - high blood pressure: if the patient has hypertension (boolean)
    - platelets: platelets in the blood (kiloplatelets/mL)
    - sex: sex of the patient (binary: 0 = female, 1 = male)
    - serum creatinine: level of serum creatinine in the blood (mg/dL)
    - serum sodium: level of serum sodium in the blood (mEq/L)
    - smoking: if the patient smokes or not (boolean)
    - time: follow-up period (days)
    - DEATH_EVENT: if the patient died during the follow-up period (boolean: 0 = survived, 1 = died)
### Correlation 
![Model Plot](images/Correlation_Heatmap.png)

### Pairplot of heart failure 
![Model Plot](images/pair_plot_heart_failure_top.png) 

### EDA Observation
    - Older age -> higher mortality	Aging hearts less resilient
    - Low ejection fraction -> key risk	Direct measure of heart pumping strength
    - High serum creatinine -> higher mortality	Indicates kidney dysfunction
    - Low serum sodium -> higher mortality	Sign of heart failure severity
    - Follow-up time inversely correlated	Death tends to occur earlier for  severe cases
    - Lifestyle factors (smoking, sex, diabetes)	Weak correlation here
### Key prediction of death 
    - Low ejection_fraction, high serum_creatinine, old age, low serum_sodium, and shorter time.
   