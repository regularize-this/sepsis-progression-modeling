import numpy as np
import pandas as pd
from data_processing import FEATURES, TREATMENT_FLAGS

STATE_NAMES  = ["Normal", "Sepsis", "Severe Sepsis", "Septic Shock"]
N_STATES = 4
ALERT_THRESHOLD = 0.30

INTERVENTIONS = {
    "oxygenFlag"      : {"name": "Oxygen",       "emoji": "🫁", "targets": ["SPO2", "HR"]},
    "antibioticsFlag" : {"name": "Antibiotics",  "emoji": "💊", "targets": ["WBC", "Lactate"]},
    "cultureFlag"     : {"name": "Blood Culture","emoji": "🧫", "targets": ["WBC"]},
    "vasoFlag"        : {"name": "Vasopressors", "emoji": "💉", "targets": ["MAP", "HR"]},
}

VITAL_FEATURES = ["HR", "SPO2", "WBC", "Lactate", "MAP"]

VITAL_THRESHOLDS = {
    "HR"     : {"normal_range": (60, 90), "bad_direction": "high"},
    "SPO2"   : {"normal_range": (90, 94), "bad_direction": "both"},
    "WBC"    : {"normal_range": (4,   12), "bad_direction": "both"},
    "Lactate": {"normal_range": (0,    2), "bad_direction": "high"},
    "MAP"    : {"normal_range": (70,  90), "bad_direction": "low"},
}


def evaluate_model(model, data_dict):
    scaler = data_dict["scaler"]
    scaled_train = data_dict["scaled_train"]
    scaled_test = data_dict["scaled_test"]
    test_lens = data_dict["test_lens"]
    
    all_states_train = [model.predict(seq) for seq in scaled_train]
    
    # Sort model states by mean Heart Rate
    hr_idx = FEATURES.index("HR")
    unscaled_means = scaler.inverse_transform(model.means_)
    
    lactate_idx = FEATURES.index("Lactate")
    lactate_order = np.argsort(unscaled_means[:, lactate_idx])
    remap = {old: new for new, old in enumerate(lactate_order)}
    
    state_labels = {v: STATE_NAMES[k] for k, v in remap.items()}
    print("\nState mapping (sorted by mean Lactate — LOW to HIGH = Normal → Septic Shock):")
    print("  ⚠️  NOTE: States are learned clusters. Validate against known patient outcomes.\n")
    for model_state, clin_name in state_labels.items():
        print(f"  HMM state {model_state}  →  {clin_name:15s}  mean Lactate = {unscaled_means[model_state, lactate_idx]:.2f}")
        
    print("\n── Learned Transition Matrix ──")
    trans_df = pd.DataFrame(
        model.transmat_,
        index=[f"From: {state_labels[i]}" for i in range(N_STATES)],
        columns=[f"To: {state_labels[j]}" for j in range(N_STATES)],
    )
    print(trans_df.round(3))
    
    print("\n── Learned Emission Means (unscaled) ──")
    means_df = pd.DataFrame(
        unscaled_means,
        index=[state_labels[i] for i in range(N_STATES)],
        columns=FEATURES,
    )
    print(means_df.round(2))
    
    print("\n── Test Set Log-Likelihood ──")
    test_scores = [model.score(seq) for seq in scaled_test]
    print(f"  Mean log-likelihood per sequence : {np.mean(test_scores):.2f}")
    print(f"  Per-timestep (avg seq length {np.mean(test_lens):.0f})  : {np.mean([s/l for s, l in zip(test_scores, test_lens)]):.4f}")
    
    return all_states_train, state_labels, remap

def learn_intervention_patterns(train_seqs_full, train_pids, all_states_train, state_labels):
    ALL_FEATURES = FEATURES + TREATMENT_FLAGS
    records = []
    for seq_full, pid, state_path in zip(train_seqs_full, train_pids, all_states_train):
        for t, (obs, model_state) in enumerate(zip(seq_full, state_path)):
            clin_state = state_labels[model_state]
            row = {"state": clin_state}
            for i, feat in enumerate(ALL_FEATURES):
                row[feat] = obs[i]
            records.append(row)

    obs_df = pd.DataFrame(records)

    patterns = {}
    for state_name in STATE_NAMES:
        patterns[state_name] = {}
        state_rows = obs_df[obs_df["state"] == state_name]
        if len(state_rows) == 0:
            continue
        for flag, info in INTERVENTIONS.items():
            with_tx    = state_rows[state_rows[flag] >= 0.5]
            without_tx = state_rows[state_rows[flag] <  0.5]
            patterns[state_name][flag] = {
                "with"      : {v: with_tx[v].mean()    for v in VITAL_FEATURES} if len(with_tx)    else None,
                "without"   : {v: without_tx[v].mean() for v in VITAL_FEATURES} if len(without_tx) else None,
                "n_with"    : len(with_tx),
                "n_without" : len(without_tx),
            }
            
    print("\n── Intervention Patterns Learned (training data) ──")
    print("  ⚠️  Observational associations — not causal estimates\n")
    for state_name in STATE_NAMES:
        if state_name not in patterns:
            continue
        pat = patterns[state_name]
        if not pat:
            continue
        print(f"  [{state_name}]")
        for flag, info in INTERVENTIONS.items():
            p = pat.get(flag, {})
            if not p or p["with"] is None or p["without"] is None:
                continue
            targets = info["targets"]
            diffs   = {v: p["with"][v] - p["without"][v] for v in targets}
            diff_str = "  ".join([f"{v}: {d:+.1f}" for v, d in diffs.items()])
            print(f"    {info['emoji']} {info['name']:15s}  n_with={p['n_with']:4d}  n_without={p['n_without']:4d}  vital diffs (with−without): {diff_str}")
        print()
        
    return patterns

def get_intervention_alerts(current_vitals, state_name, intervention_patterns):
    alerts = []
    pat = intervention_patterns.get(state_name, {})

    for flag, info in INTERVENTIONS.items():
        p = pat.get(flag)
        if not p or p["with"] is None or p["without"] is None:
            continue
        if p["n_with"] < 3: 
            continue

        reasons = []
        max_gap = 0.0

        for vital in info["targets"]:
            thresh  = VITAL_THRESHOLDS[vital]
            lo, hi  = thresh["normal_range"]
            cur_val = current_vitals.get(vital)
            if cur_val is None:
                continue

            is_abnormal = (thresh["bad_direction"] == "high" and cur_val > hi) or \
                          (thresh["bad_direction"] == "low"  and cur_val < lo)

            if not is_abnormal:
                continue

            val_with    = p["with"][vital]
            val_without = p["without"][vital]
            gap = abs(val_with - val_without)

            if thresh["bad_direction"] == "high":
                treatment_helps = val_with < val_without
            else:
                treatment_helps = val_with > val_without

            if treatment_helps and gap > 0.5:
                reasons.append({
                    "vital"      : vital,
                    "current"    : round(cur_val, 1),
                    "with_tx"    : round(val_with, 1),
                    "without_tx" : round(val_without, 1),
                    "gap"        : round(gap, 1),
                })
                max_gap = max(max_gap, gap)

        if reasons:
            alerts.append({
                "flag"      : flag,
                "name"      : info["name"],
                "emoji"     : info["emoji"],
                "reasons"   : reasons,
                "urgency"   : max_gap,
                "n_with"    : p["n_with"],
            })

    alerts.sort(key=lambda x: x["urgency"], reverse=True)
    return alerts

def progression_probability(model, seq_raw, seq_scaled, subject_id, split_label, state_labels, severity_rank, intervention_patterns):
    posteriors = model.predict_proba(seq_scaled)
    viterbi    = model.predict(seq_scaled)

    T    = len(viterbi)
    rows = []

    for t in range(T):
        curr_model_state = viterbi[t]
        curr_clin_name   = state_labels[curr_model_state]
        curr_severity    = severity_rank[curr_model_state]

        gamma_t   = posteriors[t]
        next_dist = gamma_t @ model.transmat_

        p_progress = sum(
            next_dist[k]
            for k in range(N_STATES)
            if severity_rank[k] > curr_severity
        )

        current_vitals = {feat: seq_raw[t][i] for i, feat in enumerate(FEATURES)}
        tx_alerts = get_intervention_alerts(current_vitals, curr_clin_name, intervention_patterns)

        if p_progress >= ALERT_THRESHOLD and tx_alerts:
            alert_parts = []
            for a in tx_alerts:
                reasons_str = ", ".join(
                    f"{r['vital']}={r['current']}→{r['with_tx']}"
                    for r in a["reasons"]
                )
                alert_parts.append(f"{a['emoji']} {a['name']} ({reasons_str})")
            alert_str = " | ".join(alert_parts)
        elif p_progress >= ALERT_THRESHOLD:
            alert_str = "⚠️ ESCALATE — no specific intervention pattern found"
        else:
            alert_str = ""

        row = {
            "subject_id"        : subject_id,
            "split"             : split_label,
            "hour"              : t + 1,
            "current_state"     : curr_clin_name,
            "current_severity"  : curr_severity,
        }

        for i, feat in enumerate(FEATURES):
            row[feat] = round(seq_raw[t][i], 2)

        for k in range(N_STATES):
            row[f"P({state_labels[k]})_%"] = round(next_dist[k] * 100, 2)

        row["P_progression_%"]     = round(p_progress * 100, 2)
        row["P_stay_or_improve_%"] = round((1 - p_progress) * 100, 2)
        row["interventions"]       = alert_str

        for flag, info in INTERVENTIONS.items():
            matched = next((a for a in tx_alerts if a["flag"] == flag), None)
            row[f"recommend_{info['name'].replace(' ','_')}"] = 1 if matched else 0

        rows.append(row)

    return pd.DataFrame(rows)


def run_evaluation_pipeline(model, data_dict, original_data_path):
    all_states_train, state_labels, remap = evaluate_model(model, data_dict)
    severity_rank = {model_state: remap[model_state] for model_state in range(N_STATES)}
    
    intervention_patterns = learn_intervention_patterns(
        data_dict["train_seqs_full"], data_dict["train_pids"], all_states_train, state_labels
    )
    
    print("\nComputing per-hour progression probabilities + intervention alerts...")
    all_results = []
    
    for seq_raw, seq_scaled, pid in zip(data_dict["train_seqs"], data_dict["scaled_train"], data_dict["train_pids"]):
        all_results.append(progression_probability(model, seq_raw, seq_scaled, pid, "train", state_labels, severity_rank, intervention_patterns))
        
    for seq_raw, seq_scaled, pid in zip(data_dict["test_seqs"], data_dict["scaled_test"], data_dict["test_pids"]):
        all_results.append(progression_probability(model, seq_raw, seq_scaled, pid, "test", state_labels, severity_rank, intervention_patterns))
        
    results_df = pd.concat(all_results, ignore_index=True)
    results_df["row_num"] = results_df.groupby("subject_id").cumcount() + 1
    
    alerts = results_df[results_df["interventions"] != ""]
    print(f"\n── Alert Summary (P(progression) ≥ {ALERT_THRESHOLD*100:.0f}%) ──")
    print(f"  Total hours with alerts     : {len(alerts):,}")
    print(f"  Patients with ≥1 alert      : {alerts['subject_id'].nunique()}")

    print("\n── Most Recommended Interventions ──")
    for flag, info in INTERVENTIONS.items():
        col = f"recommend_{info['name'].replace(' ','_')}"
        if col in results_df.columns:
            n = results_df[col].sum()
            print(f"  {info['emoji']} {info['name']:15s}: {int(n):,} patient-hours")

    out_path = "sepsis_hmm_progression.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}  ({len(results_df):,} rows)")
    
    df_to_merge = pd.read_csv(original_data_path)
    df_to_merge["hour"] = df_to_merge.groupby("subject_id").cumcount()+1
    results_df_csv = results_df.merge(df_to_merge, on=["subject_id", "hour"], how="left")
    
    out_path_vitals = "sepsis_hmm_progression_with_vitals.csv"
    results_df_csv.to_csv(out_path_vitals, index=False)
    print(f"Saved → {out_path_vitals}")
