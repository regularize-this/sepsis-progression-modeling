# Bayesian Modeling of Sepsis Progression in ICU Patients using MIMIC-IV Data and Bayesian Modeling

## Introduction [1][2][3]
Sepsis is the body's extreme response to an infection. It is a life-threatening medical emergency since the infection can cause a chain-reaction leading to tissue damage, organ failure and death.
Treatment includes antibiotics, IV fluids, vasopressor, supportive care or surgery if it is extreme.

The Three Stages of Sepsis:

0) SIRS
1) Sepsis
2) Severe Sepsis
3) Septic Shock

Sepsis patients do not follow a linear path and fluctuate between somewhat hidden states (SIRS → Sepsis → Severe Sepsis → Septic Shock → Recovery/Death)

Sepsis is also the leading cause of death in hospital Intensive Care Units (ICUs) and sepsis related hospitalizations increased by 40% from 2016–2021, totaling 2.5 million inpatient stays in 2021. Annual hospital costs reached $52.1 billion in 2021 (14% of all US hospital costs), over half of sepsis hospitalizations were for adults 65 years and older, and one in six older patients with sepsis died in the hospital in 2021.

## Clinical Progression of Sepsis Example
Diagram created by the authors using Google Slides icons and shapes.

![Clinical Pathway](figures/clinical_pathway_progression.png)

Patients typically progress through stages including early infection, systemic inflammatory response syndrome (SIRS), sepsis, and septic shock. Our models aim to infer these latent stages from observed clinical measurements.
## The Critical Need For Sepsis Modeling (Probelm Statement)
Timing is key with sepsis treatment; every hour that passes without treatment significantly increases the risk of permanent organ damage or death. Modeling Sepsis progression will allow for more accurate and faster diagnoses, and will help busy ICUs staff stay one step ahead (proactive treatment instead of reactive treatment). This project models sepsis progression using Bayesian approaches to infer latent clinical states.


## Features
- Feature 1
- Feature 2
- Feature 3

## How It Works

### Model A: 

Uses the following vitals and interventions to predict Sepsis states:
Heart Rate (HR)
Peripheral capillary oxygen saturation (SpO2)
White Blood Cell Count (WBC)
Lactate Levels (Lactate)
Man Arterial Pressure (MAP)
Oxygen
Antibiotics
Cultures
Vasopressors


#### model_1.py
Set initial transition probability matrix, emission means matrix, and hidden state starting probabilities (all based on researched metrics). 

Use hmmlearn (4 state Gaussian Hidden Markov Model). 


#### train_model_1.py

Train the model. Use Hmmlearn


#### evaluation.py
Generate missing flags for vitals. Mask if missing after two hours.
Use lactate levels to assign hidden sepsis states. We generate an alert whenever the probability of progressing to the next stage is above 30%.


Code exists to evaluate model (print out learned transition and emission matrices) and to generate useful charts for individual patients.

---
## Model B: Bayesian Continuous-Time Hidden Markov Model

Model B uses a **Bayesian continuous-time Hidden Markov Model (HMM)** to infer latent sepsis severity trajectories from ICU patient data. Unlike simpler baseline approaches, Model B preserves irregular ICU timing and explicitly models how treatment and elapsed time influence transitions between hidden severity states.

### Inputs

Model B uses the following observed clinical features:

- Heart Rate (HR)
- Systolic Blood Pressure (SBP)
- Mean Arterial Pressure (MAP)
- Temperature
- Respiratory Rate (RR)
- Oxygen Saturation (SpO2)
- Lactate
- White Blood Cell Count (WBC)
- Platelets
- Creatinine
- Bilirubin
- Hemoglobin
- SOFA Score
- Elapsed time between observations (`delta_t`)

It also incorporates treatment indicators as transition covariates:

- Antibiotics
- IV Fluids
- Vasopressors

### Core Idea

Model B assumes that septic ICU patients move through a small number of **hidden clinical severity states** over time. These states are not directly observed, but are inferred from the patient’s vital signs, laboratory values, organ dysfunction measures, and treatment history.

The model is designed to answer questions such as:

- What latent severity state is the patient currently in?
- How likely is the patient to worsen or improve next?
- How do active treatments affect transition tendencies?
- How much risk increases if treatment is delayed?

### Why Model B Is Different

Model B extends beyond a standard Gaussian HMM in several important ways:

- **Bayesian framework**  
  Captures uncertainty in model parameters and state assignments.

- **Continuous-time awareness**  
  Preserves irregular ICU timestamps instead of forcing data into fixed hourly bins.

- **Treatment-aware transitions**  
  Transition probabilities depend on active treatment exposure.

- **Severity-aware transitions**  
  Baseline SOFA is included to reflect the patient’s initial illness burden.

- **Time-aware transitions**  
  Elapsed time (`delta_t`) influences the likelihood of moving between hidden states.

### Transition Structure

For each current state and next state, transition probabilities are modeled as a function of:

- a baseline transition term,
- treatment exposure,
- treatment × baseline SOFA interaction,
- baseline severity,
- elapsed time (`delta_t`).

These transition logits are then passed through a row-wise softmax to produce valid transition probabilities.

### Inference

The model is implemented in **PyMC** and trained using **Automatic Differentiation Variational Inference (ADVI)**.

Training setup:

- 50,000 ADVI iterations
- 2,000 posterior samples drawn from the variational approximation

This makes the Bayesian HMM computationally feasible for long ICU trajectories while still preserving posterior uncertainty.

### Outputs

Model B produces:

- patient-level latent state probabilities,
- learned severity state profiles,
- treatment-conditioned transition summaries,
- short-horizon trajectory forecasts,
- decision-support outputs for treatment timing and progression risk.

### Interpretation of Hidden States

The learned hidden states are interpreted post hoc as clinically meaningful severity profiles such as:

- Mild
- Moderate
- Shock-dominant

These labels are assigned after training based on organ dysfunction patterns and clinical feature profiles, rather than being hard-coded in advance.

### Decision Support Use Case

Model B is designed not only for retrospective trajectory analysis, but also for **clinical decision support simulation**. It can be used to:

- compare treatment scenarios,
- estimate worsening risk,
- evaluate the effect of delayed intervention,
- identify patients approaching more severe latent states.

### Model B Notebooks

- [Data Extraction](./notebooks/01_Data_Extraction.ipynb)
- [Data Preprocessing](./notebooks/02_Data_Preprocessing.ipynb)
- [Exploratory Data Analysis](./notebooks/03_EDA.ipynb)
- [SOFA Feature Engineering](./notebooks/04_SOFA_Feature_Engineering.ipynb)
- [Bayesian HMM Training](./notebooks/05_Bayesian_HMM_ADVI.ipynb)
- [Prediction and DSS](./notebooks/06_Prediction_DSS.ipynb)
---



## Installation
git clone https://github.com/username/sepsis-progression-model
cd sepsis-progression-model
pip install -r requirements.txt
python src/train_model_1.py

## Usage
Example commands or screenshots.

## Results
What results you got.

## Future Improvements
Possible next steps

### Collaborators
Ryan Abdelrahim, Collin Kim, Yuta Kobayashi, Junghoon Yum and Nick Thompson

### References
[1] https://www.cdc.gov/sepsis/about/index.html#:~:text=Sepsis%20is%20a%20life%2Dthreatening%20medical%20emergency%20that,infections%2C%20such%20as%20influenza%20*%20Fungal%20infections
[2] https://my.clevelandclinic.org/health/diseases/12361-sepsis
[3] https://www.salvilaw.com/blog/sepsis-stages/
