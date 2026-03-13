# Dataset

## Data Source: MIMIC-IV ICU database*
- De-identified Electronic Health Records data from Beth Israel Deaconess Medical Center (2008–2022)
- Data on 257,000+ distinct patients, 524,000+ hospital admissions, and 73,000 ICU stays
- ICU data contains time series information about vitals (heart rate, respiration rate, etc), lab tests/results (white blood cell counts, lactate, etc), and treatments (antibiotics administered, etc)
- Allows us to 'replay' tens of thousands of ICU stays to see if a ML model could have spotted sepsis earlier

## Cohort Selection
- Adult patients (18-65)
  - Want to avoid pediatric physiology
  - Want to reduce the chances of age related comorbidities affecting the model
- Evidence of infecction:
  -  Patients who were suspected of having sepsis are defined as patients who were administered antibiotics and had labs ordered for them within an hour [7]**
-  First ICU stay:
  - Want model to learn from as many patients as possible so this prevents duplicate patients in the data
- ICU stay length between 12 hours and 10 days
  - 12 hours is the minimum to allow enough time for data collection
  - 10 days is the average ICU stay and above that may be related to patients with other complications[8] 


## Final Cohort
N = 1503 patients

### Variables Used and Reasoning
- Labs:
- Vitals:
- Interventions:

*Data access requires an approval process via PhysioNet (https://physionet.org/content/mimiciv/)
**Sepsis related labs and antibiotics are as follows:

Cultures used for sepsis evaluation (chosen because they help identify the source of sepsis in an acute evaluation): [1], [2], [3], [4], [5]
- Blood
- Urinary
- Sputum
- Wound
- CSF
- PAN (culture test that includes a pair of blood cultures, urinalysis with culture and sensitivity, sputum sample if a productive cough is present), and a chest x-ray to rule out pneumonia)
- BAL
- PFC

Antibiotics used in the treatment of sepsis: [6]

-Vancomycin

-Ampicillin

-Azithromycin

-Aztreonam

-Cefazolin

-Cefepime

-Ceftriaxone

-Clindamycin

-Levofloxacin

-Linezolid

-Meropenem

-Metronidazole

-Moxifloxacin

-Piperacillin

-Piperacillin/Tazobactam (Zosyn)

-Bactrim (SMX/TMP)

-Tobramycin

-Ertapenem sodium (Invanz)



### References:
[1] https://my.clevelandclinic.org/health/diagnostics/22155-bacteria-culture-test

[2] https://www.sepsis.org/sepsis-basics/testing-for-sepsis/

[3] https://medlineplus.gov/lab-tests/bronchoscopy-and-bronchoalveolar-lavage-bal/

[4] https://accessmedicine.mhmedical.com/content.aspx?bookid=365&sectionid=43074916

[5] https://medlineplus.gov/ency/article/003725.htm

[6] https://med.stanford.edu/content/dam/sm/bugsanddrugs/documents/clinicalpathways/SHC-Sepsis-ABX-Guide.pdf

[7] Evans L, Rhodes A, Alhazzani W, et al. Surviving Sepsis Campaign: international guidelines for management of sepsis and septic shock: 2021. Crit Care Med. 2021 Nov 1;49(11):e1063-e1143.

[8] https://hcup-us.ahrq.gov/reports/statbriefs/sb306-overview-sepsis-2016-2021.pdf
