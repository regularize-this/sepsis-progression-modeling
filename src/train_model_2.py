from data_processing import load_and_clean_data, build_sequences, split_and_scale_data
from model import build_hmm
from evaluation import run_evaluation_pipeline

DATA_PATH = "final_patient_df.csv"
N_STATES = 4
N_ITER = 100
RANDOM_SEED = 10

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
