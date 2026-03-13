import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import from your other modules
from model import build_hmm
from evaluation import run_evaluation_pipeline

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
DATA_PATH = "final_patient_df.csv"
FEATURES = ["HR", "SPO2", "WBC", "Lactate", "MAP"]
TREATMENT_FLAGS = ["oxygenFlag", "antibioticsFlag", "cultureFlag", "vasoFlag"]
MISSING_FLAGS = ["Lactate_missing", "WBC_missing"]
MASK_AFTER_N_HOURS = 2

# Model parameters
N_STATES = 4
N_ITER = 100
RANDOM_SEED = 10


# ==========================================
# DATA PROCESSING FUNCTIONS
# ==========================================
def load_and_clean_data(data_path):
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df):,}  |  Patients: {df['subject_id'].nunique():,}")

    # Impute missing values with column median
    for col in FEATURES + TREATMENT_FLAGS:
        df[col] = df[col].fillna(df[col].median())

    # Clip physiological outliers
    df["HR"]      = df["HR"].clip(0, 250)
    df["SPO2"]    = df["SPO2"].clip(50, 100)
    df["WBC"]     = df["WBC"].clip(0, 50)
    df["Lactate"] = df["Lactate"].clip(0, 20)
    df["MAP"]     = df["MAP"].clip(0, 200)

    # Binary flags -> 0/1
    for flag in TREATMENT_FLAGS:
        df[flag] = (df[flag] > 0).astype(float)
        
    return df

def build_sequences(df):
    print("Building sequences...")
    lactate_idx_f = FEATURES.index("Lactate")
    wbc_idx_f     = FEATURES.index("WBC")
    
    sequences      = []
    sequences_full = []
    missing_masks  = []
    lengths        = []
    pids           = []

    for pid, group in df.groupby("subject_id", sort=False):
        arr      = group[FEATURES].values.astype(np.float64)
        arr_full = group[FEATURES + TREATMENT_FLAGS].values.astype(np.float64)

        # Build a (T, n_features) boolean mask — True = value is a stale forward-fill
        T    = len(arr)
        mask = np.zeros((T, len(FEATURES)), dtype=bool)

        for feat, feat_idx in [("Lactate", lactate_idx_f), ("WBC", wbc_idx_f)]:
            missing_col = f"{feat}_missing"
            if missing_col in group.columns:
                missing_flags = group[missing_col].values
                consec = 0
                for t in range(T):
                    if missing_flags[t] == 1:
                        consec += 1
                    else:
                        consec = 0
                    # Mask if consecutively missing for more than threshold
                    if consec > MASK_AFTER_N_HOURS:
                        mask[t, feat_idx] = True

        if T >= 3:
            sequences.append(arr)
            sequences_full.append(arr_full)
            missing_masks.append(mask)
            lengths.append(T)
            pids.append(pid)
            
    print(f"  Sequences: {len(sequences):,}  |  Avg length: {np.mean(lengths):.1f}  |  Max: {max(lengths)}")
    return sequences, sequences_full, missing_masks, lengths, pids

def split_and_scale_data(sequences, sequences_full, missing_masks, lengths, pids, n_sample=1503, split_ratio=0.8, random_seed=10):
    np.random.seed(random_seed)
    
    sample_idx = np.random.choice(len(sequences), size=n_sample, replace=False)
    
    sequences = [sequences[i] for i in sample_idx]
    sequences_full = [sequences_full[i] for i in sample_idx]
    missing_masks = [missing_masks[i] for i in sample_idx]
    lengths = [lengths[i] for i in sample_idx]
    pids = [pids[i] for i in sample_idx]
    
    print(f"  Sampled {n_sample} patients for training/testing.")
    
    # 80/20 Train/Test Split
    split = int(split_ratio * n_sample)
    train_seqs       = sequences[:split]
    train_seqs_full  = sequences_full[:split]
    train_masks      = missing_masks[:split]
    train_lens       = lengths[:split]
    train_pids       = pids[:split]
    
    test_seqs        = sequences[split:]
    test_seqs_full   = sequences_full[split:]
    test_masks       = missing_masks[split:]
    test_lens        = lengths[split:]
    test_pids        = pids[split:]
    
    print(f"  Train: {len(train_seqs)} patients  |  Test: {len(test_seqs)} patients")
    
    # Normalise (fit on train only to avoid leakage)
    scaler = StandardScaler()
    scaler.fit(np.vstack(train_seqs))
    
    def scale_and_mask(seqs, masks):
        """
        Scale sequences then zero-out stale forward-filled values.
        In z-score space, 0 = feature mean
        """
        scaled = []
        for seq, mask in zip(seqs, masks):
            s = scaler.transform(seq).copy()
            s[mask] = 0.0   # mask stale Lactate/WBC with feature mean
            scaled.append(s)
        return scaled, np.vstack(scaled), [len(s) for s in scaled]

    scaled_train, X_train, train_lens = scale_and_mask(train_seqs, train_masks)
    scaled_test, X_test, test_lens = scale_and_mask(test_seqs, test_masks)
    
    total_masked = sum(m.sum() for m in train_masks)
    total_vals   = sum(len(s) * len(FEATURES) for s in train_seqs)
    print(f"  Masked {total_masked:,} stale forward-fill values "
          f"({total_masked/total_vals*100:.1f}% of train data) "
          f"[threshold: >{MASK_AFTER_N_HOURS} consecutive missing hours]")
          
    return {
        "train_seqs": train_seqs, "train_seqs_full": train_seqs_full, "train_pids": train_pids,
        "scaled_train": scaled_train, "X_train": X_train, "train_lens": train_lens,
        "test_seqs": test_seqs, "test_seqs_full": test_seqs_full, "test_pids": test_pids,
        "scaled_test": scaled_test, "X_test": X_test, "test_lens": test_lens,
        "scaler": scaler
    }


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    # 1. Load and Clean Data
    df = load_and_clean_data(DATA_PATH)
    
    # 2. Build Sequences
    sequences, sequences_full, missing_masks, lengths, pids = build_sequences(df)
    
    # 3. Split and Scale Data
    data_dict = split_and_scale_data(
        sequences, sequences_full, missing_masks, lengths, pids, 
        n_sample=1503, split_ratio=0.8, random_seed=RANDOM_SEED
    )
    
    # 4. Build HMM Model
    model = build_hmm(n_states=N_STATES, n_iter=N_ITER, random_seed=RANDOM_SEED, scaler=data_dict["scaler"])
    
    # 5. Train Model
    print(f"\nTraining Gaussian HMM ({N_STATES} states, {N_ITER} iterations)...")
    model.fit(data_dict["X_train"], data_dict["train_lens"])
    print("\nTraining complete.")
    
    # 6. Evaluate and Export Results
    run_evaluation_pipeline(model, data_dict, original_data_path=DATA_PATH)

if __name__ == "__main__":
    main()
