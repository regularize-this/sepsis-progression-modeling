import numpy as np
from hmmlearn import hmm

STATE_NAMES  = ["Normal", "Sepsis", "Severe Sepsis", "Septic Shock"]

def build_hmm(n_states=4, n_iter=100, random_seed=10, scaler=None):
    START_PROBS = np.array([0.25, 0.40, 0.22, 0.13])

    # Transitions per hour
    TRANS_PROBS = np.array([
        [0.84, 0.13, 0.02, 0.01],   # Normal
        [0.10, 0.74, 0.13, 0.03],   # Sepsis
        [0.00, 0.12, 0.70, 0.18],   # Severe Sepsis
        [0.00, 0.00, 0.12, 0.88],   # Septic Shock
    ])

    # Emission means
    # Features:       HR    SPO2   WBC   Lactate  MAP
    EMISSION_MEANS = np.array([
        [82,   98,  8.5,    1.2,   82],   
        [96,   95,   14.0,   2.1,   72],  
        [108,   92,   18.0,   3.8,  68],  
        [118,   88,   22.0,   6.5,  62],  
    ])
    
    if scaler is not None:
        EMISSION_MEANS_SCALED = scaler.transform(EMISSION_MEANS)
    else:
        EMISSION_MEANS_SCALED = EMISSION_MEANS

    # Build model and manually inject priors
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_seed,
        verbose=True,
        init_params="c",   
        params="stmc",     
    )

    model.startprob_ = START_PROBS
    model.transmat_  = TRANS_PROBS
    model.means_     = EMISSION_MEANS_SCALED

    print("  Clinical priors injected (general ICU population):")
    for name, p in zip(STATE_NAMES, START_PROBS):
        print(f"    {name:15s}: {p*100:.0f}% of ICU hours at admission")
    print()

    return model
