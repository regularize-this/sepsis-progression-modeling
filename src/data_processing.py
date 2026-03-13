!pip install duckdb

import pandas as pd
import duckdb
import numpy as np
from tqdm import tqdm

# initiate
con = duckdb.connect()

# Load in file paths
icufilepath = # add filepath to icu data - should end in /icu/'
icuStaysFilepath = icufilepath + 'icustays.csv.gz'
inputEventsFp = icufilepath + 'inputevents.csv.gz'
procedureEventsFp = icufilepath + 'procedureevents.csv.gz'
chartEventsFp = icufilepath + 'chartevents.csv.gz'

hospFilepath = # add filepath to hosp data - should end in /hosp/'
ptFilepath = hospFilepath + 'patients.csv.gz'

# get first icu stay per patient
con.execute(f"""
    CREATE OR REPLACE TABLE fp AS
    SELECT * FROM read_csv_auto('{icuStaysFilepath}');
    """)

# identifier hierarchy: subject_id → hadm_id → stay_id (patient → hospital admission → ICU stay).
# rank each patient's ICU stay by intime and keep the first
# returns distinct patients -> at 65366 patients
firstICUStay = con.execute(f"""
WITH ranked AS(
    SELECT 
      subject_id,
      hadm_id, 
      stay_id,      
      intime,
      outtime,
      first_careunit,
      ROW_NUMBER() OVER(
        PARTITION BY subject_id
        ORDER BY intime)
        AS rn
     FROM fp
)
SELECT * 
FROM ranked
WHERE rn=1
""").df()

# # get duration of stay
firstICUStay['duration'] = firstICUStay['outtime'] - firstICUStay['intime']

# only keep those who have been in for min 12 hours and max 10 days -> patients now at 58506
firstStayFilteredHours = firstICUStay[
    (firstICUStay['duration'] > pd.Timedelta('12 hours')) &
    (firstICUStay['duration'] < pd.Timedelta('10 days'))
]

# filter to adults 
# register current table of patients as DuckDB view
con.register("firstICUStay", firstStayFilteredHours)

# load in patients table (age in here)
con.execute(f"""
    CREATE OR REPLACE TABLE patients AS
    SELECT *
    FROM read_csv_auto('{ptFilepath}');
""")

# grab patient selected's ages
ages = con.execute(f"""
SELECT 
    f.*,
    p.anchor_age
FROM firstICUStay f
LEFT JOIN patients p
    USING(subject_id)
""").df()

# filter to >18 and < 65 -> now at 29421 patients 
adults = ages.loc[(ages['anchor_age'] >= 18) & (ages['anchor_age'] <= 65)]

# Filter to those who got lab cultures ordered and antibiotics administered within an hour of each other

# define list of antibiotics and cultures
# in procedureevents - which has starttime and endtime
culturesSepsis = [225401, #Blood
                  225437, # CSf
                  225444, # Pan
                  225451, # sputum
                  225454, # urine
                  225816, # wound
                  225817, # BAL
                  225818  # Pleural
                 ]

# in inputevents - which has starttime and endtime
antibioticsSepsisID =  [225798, # Vancomycin
                        225842, # Ampicillin
                        225845, # Azithromycin
                        225847, # Aztreonam
                        225850, # Cefazolin
                        225851, # Cefepime
                        225855, # Ceftriaxone
                        225860, # Clindamycin
                        225879, # Levofloxacin
                        225881, # Linezolid
                        225883, # Meropenem
                        225884, # Metronidazole
                        225886, # Moxifloxacin
                        225892, # Piperacillin
                        225893, # Piperacillin/Tazobactam (Zosyn)
                        225899, # Bactrim (SMX/TMP)
                        225902, # Tobramycin
                        229061  # Ertapenem sodium (Invanz)
                       ]

# register current group of patients as DuckDB view
con.register("adults", adults)

# get our final cohort -> Final Cohort has 1503 patients
# EPOCH converts time to seconds

cohort = duckdb.query(f"""
SELECT DISTINCT
    v.subject_id,
    v.stay_id,
    v.hadm_id,
    v.intime,
    v.outtime
FROM adults v
JOIN (
    SELECT *
    FROM read_csv_auto('{inputEventsFp}')
    WHERE itemid IN ({','.join(map(str, antibioticsSepsisID))})
) a
    ON v.subject_id = a.subject_id
   AND v.stay_id = a.stay_id
   AND a.starttime >= v.intime
   AND a.endtime <= v.outtime
JOIN (
    SELECT *
    FROM read_csv_auto('{procedureEventsFp}')
    WHERE itemid IN ({','.join(map(str, culturesSepsis))})
) c
    ON v.subject_id = c.subject_id
   AND v.stay_id = c.stay_id
   AND ABS(EXTRACT(EPOCH FROM (c.starttime - a.starttime))) <= 3600
""").to_df()

# make each patient have a time series df with the variabels we are tracking 

# define oxygen-support related items
# strongly indicate ETT or trach ventilation
invasiveOxygenSupport = [
220339,	#PEEP set
223849,	#Ventilator Mode
224700,	#Total PEEP Level
224829,	#Trach Tube Type
225308,	#Nasal ETT
225411,	#Patient on vent
225792,	#Invasive Ventilation
226260,	#Mechanically Ventilated
228719,	#Vented
229314,	#Ventilator Mode (Hamilton)
]

nonInvasiveOxygen = [
225949,	#NIV Mask
225794,	#Non-invasive Ventilation
227578,	#BiPap Mask
]

allOxy = invasiveOxygenSupport + nonInvasiveOxygen

# define vasopressors 
vasopressors = [
221289,	#Epinephrine
221653,	#Dobutamine
221662,	#Dopamine
221749,	#Phenylephrine
221906,	#Norepinephrine
221986,	#Milrinone
222315,	#Vasopressin
229617,	#Epinephrine.
229630,	#Phenylephrine (50/250)
229631,	#Phenylephrine (200/250)_OLD_1
229632,	#Phenylephrine (200/250)
229709,	#Angiotensin II (Giapreza)
229764,	#Angiotensin II (Giapreza)
]

# define labs and vitals
labsAndVitals = [
    220045, # HR
    220052, # MAP
    220277, # SPO2
    220546, # WBC
    225668 # lactic acid
]

# register our cohort as DuckDB view
con.register("cohort", cohort)

# oxygen exists in chartevents and procedureevents so grab from both 
# for each patient in the cohort grab anything oxygen related for the stay we are looking at 
oxyEventsChart = con.execute(f"""
SELECT ce.subject_id, ce.stay_id, ce.hadm_id, ce.charttime, ce.itemid, ce.valuenum AS value
FROM read_csv_auto('{chartEventsFp}') AS ce
JOIN cohort AS c
  ON ce.subject_id = c.subject_id
 AND ce.stay_id = c.stay_id
 AND ce.charttime BETWEEN c.intime AND c.outtime
 AND ce.itemid IN ({','.join(map(str, allOxy))})
""").df()

oxyEventsChart2 = con.execute(f"""
SELECT ce.subject_id, ce.stay_id, ce.hadm_id, ce.starttime, ce.itemid
FROM read_csv_auto('{procedureEventsFp}') AS ce
JOIN cohort AS c
  ON ce.subject_id = c.subject_id
 AND ce.stay_id = c.stay_id
 AND ce.starttime BETWEEN c.intime AND c.outtime
 AND ce.itemid IN ({','.join(map(str, allOxy))})
""").df()

# get vasopressor related information
vasopressorsDf = con.execute(f"""
SELECT ce.subject_id, ce.stay_id, ce.hadm_id, ce.starttime, ce.itemid
FROM read_csv_auto('{inputEventsFp}') AS ce
JOIN cohort AS c
  ON ce.subject_id = c.subject_id
 AND ce.stay_id = c.stay_id
 AND ce.starttime BETWEEN c.intime AND c.outtime
 AND ce.itemid IN ({','.join(map(str, vasopressors))})
""").df()

# get cultures related information
culturesDf = con.execute(f"""
SELECT ce.subject_id, ce.stay_id, ce.hadm_id, ce.starttime, ce.itemid
FROM read_csv_auto('{procedureEventsFp}') AS ce
JOIN cohort AS c
  ON ce.subject_id = c.subject_id
 AND ce.stay_id = c.stay_id
 AND ce.starttime BETWEEN c.intime AND c.outtime
 AND ce.itemid IN ({','.join(map(str, culturesSepsis))})
""").df()

# get chart events (labs and vitals)
chartEvents = con.execute(f"""
SELECT ce.subject_id, ce.stay_id, ce.hadm_id, ce.charttime, ce.itemid, ce.valuenum AS value
FROM read_csv_auto('{chartEventsFp}') AS ce
JOIN cohort AS c
  ON ce.subject_id = c.subject_id
 AND ce.stay_id = c.stay_id
 AND ce.charttime BETWEEN c.intime AND c.outtime
 AND ce.itemid IN ({','.join(map(str, labsAndVitals))})
""").df()

# get antibiotics related information
antibiotics = con.execute(f"""
SELECT ce.subject_id, ce.stay_id, ce.hadm_id, ce.starttime, ce.itemid
FROM read_csv_auto('{inputEventsFp}') AS ce
JOIN cohort AS c
  ON ce.subject_id = c.subject_id
 AND ce.stay_id = c.stay_id
 AND ce.starttime BETWEEN c.intime AND c.outtime
 AND ce.itemid IN ({','.join(map(str, antibioticsSepsisID))})
""").df()

# define the functions used to filter and clean the data

# for oxygen information: set flags for oxygen support type, resample to hourly and drop duplicates
def oxygenFinding(oxy1, oxy2):
    # set time index 
    oxy1 = oxy1.set_index('charttime').drop(columns = ['subject_id','stay_id', 'hadm_id','value'], axis=1).sort_index()
    oxy2 = oxy2.set_index('starttime').drop(columns = ['subject_id','stay_id', 'hadm_id'], axis=1).sort_index()

    # set flags where 2 is invasive ventilation, 1 is noninvasive and 0 is no oxygen support
    oxy1['oxygenFlag'] = np.where(oxy1['itemid'].isin(invasiveOxygenSupport), 2,
                               np.where(oxy1['itemid'].isin(nonInvasiveOxygen), 1, 0))
    oxy2['oxygenFlag'] = np.where(oxy2['itemid'].isin(invasiveOxygenSupport), 2,
                               np.where(oxy2['itemid'].isin(nonInvasiveOxygen), 1, 0))
    # resample to hourly
    oxy1 = oxy1['oxygenFlag'].resample('h').max()
    oxy2 = oxy2['oxygenFlag'].resample('h').max()

    # concat and turn to df
    dfOxy = pd.DataFrame(pd.concat([oxy1, oxy2]))

    # drop duplicates and sort
    dfOxy = dfOxy[~dfOxy.index.duplicated(keep='first')]
    dfOxy = dfOxy.sort_index()

    return dfOxy

# for antibiotics: set flag for when meds administered and resample to hourly
def antiFinding(anti):
    # set time index
    anti = anti.set_index('starttime').drop(columns = ['subject_id','stay_id', 'hadm_id'], axis=1).sort_index()

    # set flag for when meds administered
    anti['antibioticsFlag'] = np.where(anti['itemid'] > 0, 1, 0)
    anti = anti.drop(['itemid'], axis=1)

    # resample to hourly
    anti = anti.resample('h').max()
    
    return anti

# for vitals and labs, resampel to hourly and take the mean value if there are many
def vitalsAndlabsFixing(tester):
    # set time index
    tester = tester.set_index('charttime').drop(columns = ['subject_id','stay_id', 'hadm_id'], axis =1).sort_index()

    # define e/ lab and vital
    HR = tester[tester['itemid'] == 220045].resample('h').mean()
    MAP = tester[tester['itemid'] == 220052].resample('h').mean()
    SPO2 = tester[tester['itemid'] == 220277].resample('h').mean()
    WBC = tester[tester['itemid'] == 220546].resample('h').mean()
    Lactic = tester[tester['itemid'] == 225668].resample('h').mean()

    # combine all
    combined = (
    HR[['value']].rename(columns={'value': 'HR'})
    .join(MAP[['value']].rename(columns={'value': 'MAP'}), how="outer")
    .join(SPO2[['value']].rename(columns={'value': 'SPO2'}), how="outer")
    .join(WBC[['value']].rename(columns={'value': 'WBC'}), how="outer")
    .join(Lactic[['value']].rename(columns={'value': 'Lactate'}), how="outer")
    )
    
    return combined

# set a flag for when cultures were ordered and resampel to hourly taking the maximum value
def cultureFinding(df):
    # set time index
    df = df.set_index('starttime').drop(columns = ['subject_id','stay_id', 'hadm_id'], axis=1).sort_index()

    # set flag for when cultures ordered
    df['cultureFlag'] = np.where(df['itemid'] > 0, 1, 0)
    df = df.drop(['itemid'], axis=1)

    # resample to hourly
    df = df.resample('h').max()

    return df

# set flag for when vasopressors administered and resample to hourly
def vasopFinding(df):
    # set time index
    df = df.set_index('starttime').drop(columns = ['subject_id','stay_id', 'hadm_id'], axis=1).sort_index()

    # set flag for when vasopressors administered
    df['vasoFlag'] = np.where(df['itemid'] > 0, 1, 0)
    df = df.drop(['itemid'], axis=1)

    # resample to hourly
    df = df.resample('h').max()

    return df

# clean up all the information per patient
def cleanUp(df):

    ## VITALS AND LABS
    # forward and back fill vitals 
    df[['HR', 'MAP', 'SPO2']] = df[['HR', 'MAP', 'SPO2']].ffill().bfill()
    
    # Labs are sparse and clinician‑triggered.
    # Forward‑filling indefinitely implies stability that isn’t clinically true.
    # A capped window respects the physiology and the sampling process.
    
    df['WBC'] = df['WBC'].ffill(limit=12) # WBC done with routine icu cbc panels which are done every 12 or 24 hours
    df['Lactate'] = df['Lactate'].ffill(limit=6) # fill till 6 hours (normally done every 4-6 hours) 
    
    # for the reamining values, adding a missing indicator first then filling in with the mean
    # simple statistical imputations are not physiologically accurate, but they are mathematically stable and ensure model compatibility
    
    for col in ['WBC', 'Lactate']:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        
    df['WBC'] = df['WBC'].fillna(df['WBC'].mean())
    df['Lactate'] = df['Lactate'].fillna(df['Lactate'].mean())

    ## OXYGEN SUPPORT
    # assume ICU charting happens every 2 hours - DOI: 10.4037/ccn2010406
    
    # Forward fill up to 2 hours (limit=2)
    df['oxygenFlag'] = df['oxygenFlag'].ffill(limit=2)
    
    # Remaining NaNs = no oxygen
    df['oxygenFlag'] = df['oxygenFlag'].fillna(0)

    ## VASOPRESSORS 
    df['vasoFlag'] = df['vasoFlag'] .fillna(0)

    ## ANTIBIOTICS 
    df['antibioticsFlag'] = df['antibioticsFlag'] .fillna(0)

    ## CULTURES 
    df['cultureFlag'] = df['cultureFlag'] .fillna(0)

    return df

# grab, filter and process all variables per patient
patientsDfs =[]
ids = cohort['subject_id']

for id in tqdm(ids, desc="Processing patients"):

    # filter all tables to subject id
    ce = chartEvents[chartEvents['subject_id'] == id] # vitals and labs
    oxy1 = oxyEventsChart[oxyEventsChart['subject_id'] == id] # chart events oxygen
    oxy2 = oxyEventsChart2[oxyEventsChart2['subject_id'] == id] # procedure events oxygen
    anti = antibiotics[antibiotics['subject_id'] == id] # antibiotics
    culture = culturesDf[culturesDf['subject_id'] == id] # cultures
    vaso = vasopressorsDf[vasopressorsDf['subject_id'] == id] # vasopressors
    
    # set e/ to time index and hourly sampling + cleaning + flag setting
    vitalsLabs = vitalsAndlabsFixing(ce)
    oxygen = oxygenFinding(oxy1, oxy2)
    antibio = antiFinding(anti)
    cultures = cultureFinding(culture)
    vasop = vasopFinding(vaso)
    
    # get timeline for pt
    row = firstICUStay[firstICUStay['subject_id'] == id]
    hour_index = pd.date_range(
        start = pd.to_datetime(row['intime'].iloc[0]).floor('h'), # round to nearest whole hour
        end = pd.to_datetime(row['outtime'].iloc[0]).ceil('h'),
        freq = 'h')
    
    # make final df
    # get full timeline of patient stay
    master_df = pd.DataFrame({"time": hour_index})
    master_df.set_index("time", inplace=True)
    # merge all df 
    merged = pd.concat([master_df, vitalsLabs, oxygen, antibio, cultures, vasop], axis=1)
    
    # clean up all columns
    dfFinal = cleanUp(merged)
    dfFinal['SubjectId'] = id
    
    patientsDfs.append(dfFinal)
