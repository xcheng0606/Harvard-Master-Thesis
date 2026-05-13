"""
H-AIPW with LLM as external data source.
Requires: llm_predictions/predictions_claude.csv (from llm_predict.py)
"""

import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "two paper" / "IST3 dataset" / "datashare_aug2015.sas7bdat"
LLM_PATH  = Path(__file__).parent / "llm_predictions" / "predictions_claude.csv"
N_SPLITS  = 30
N_REPS    = 1000
SAMPLE_SIZES = [100, 200, 344]   # n=688 & 1376 not feasible given pool size (n=389)
SEED      = 42
REF_IST3_DM = None   # set at runtime

LR_KWARGS = dict(max_iter=2000, solver="lbfgs", random_state=SEED)

def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(**LR_KWARGS))

COVARIATE_COLS = [
    "age", "gender", "randdelay", "sbprand", "nihss",
    "gcs_score_rand", "atrialfib_rand", "stroketype", "weight", "glucose",
    "weakface_rand", "weakarm_rand", "weakleg_rand", "dysphasia_rand",
    "hemianopia_rand", "visuospat_rand", "brainstemsigns_rand", "otherdeficit_rand",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ist3():
    print("Loading IST3 data...")
    df, _ = pyreadstat.read_sas7bdat(str(DATA_PATH))
    required = ["itt_treat", "ohs6"] + COVARIATE_COLS
    df = df.dropna(subset=required).reset_index(drop=True)
    df["outcome"] = (df["ohs6"] <= 2).astype(float)   # OHS<=2: alive and independent
    df["treat"]   = df["itt_treat"].astype(float)
    df["pat_id"]  = df.index
    print(f"  Complete-case n={len(df)}, outcome rate={df.outcome.mean():.3f}")
    return df


def load_llm_predictions(df):
    print(f"Loading LLM predictions from {LLM_PATH}...")
    llm = pd.read_csv(LLM_PATH).dropna(subset=["predicted_prob"])
    llm_wide = llm.pivot(index="pat_id", columns="condition", values="predicted_prob")
    llm_wide.columns = ["f3_0", "f3_1"]
    llm_wide = llm_wide.reset_index()
    df = df.merge(llm_wide, on="pat_id", how="left")
    n_with = df[["f3_0", "f3_1"]].notna().all(axis=1).sum()
    print(f"  Patients with LLM predictions: {n_with} / {len(df)}")
    return df


# ── Arm-specific influence functions ──────────────────────────────────────────

def phi1(Y, A, Q1, pi):
    """Treatment-arm influence function: estimates E[Y(1)]."""
    return (A / pi) * (Y - Q1) + Q1

def phi0(Y, A, Q0, pi):
    """Control-arm influence function: estimates E[Y(0)]."""
    return ((1 - A) / (1 - pi)) * (Y - Q0) + Q0


# ── Internal cross-fitted model ────────────────────────────────────────────────

def crossfit_arm_models(Y, A, X, n_splits=N_SPLITS):
    """Logistic regression with cross-fitting. Returns Q1_hat, Q0_hat."""
    n = len(Y)
    Q1_hat = np.zeros(n)
    Q0_hat = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        mask1 = A[tr] == 1
        mask0 = A[tr] == 0
        if mask1.sum() >= 2:
            m1 = make_lr().fit(X[tr][mask1], Y[tr][mask1])
            Q1_hat[te] = m1.predict_proba(X[te])[:, 1]
        if mask0.sum() >= 2:
            m0 = make_lr().fit(X[tr][mask0], Y[tr][mask0])
            Q0_hat[te] = m0.predict_proba(X[te])[:, 1]
    return Q1_hat, Q0_hat


# ── H-AIPW arm combination ────────────────────────────────────────────────────

def haipw_arm(phi_list):
    """Optimal combination of arm-specific IFs. Returns (theta_a, scaled_var_a, lam_a)."""
    Psi = np.column_stack(phi_list)
    if Psi.shape[1] == 1:
        return float(Psi.mean()), float(np.var(Psi[:, 0], ddof=1)), np.array([1.0])
    Sigma = np.cov(Psi.T)
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma += 1e-6 * np.eye(Sigma.shape[0])
        Sigma_inv = np.linalg.inv(Sigma)
    ones = np.ones(Psi.shape[1])
    lam  = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)
    return float(lam @ Psi.mean(axis=0)), float(lam @ Sigma @ lam), lam


def haipw_ate(phi1_list, phi0_list):
    """H-AIPW ATE = θ̂₁ − θ̂₀. Returns (ATE, scaled_var, lam1, lam0)."""
    theta1, sv1, lam1 = haipw_arm(phi1_list)
    theta0, sv0, lam0 = haipw_arm(phi0_list)
    Psi1 = np.column_stack(phi1_list) @ lam1
    Psi0 = np.column_stack(phi0_list) @ lam0
    cov01 = float(np.cov(Psi1, Psi0)[0, 1])
    return theta1 - theta0, sv1 + sv0 - 2 * cov01, lam1, lam0


# ── Subsampling analysis ───────────────────────────────────────────────────────

def run_subsampling(df, sample_sizes=SAMPLE_SIZES, n_reps=N_REPS, ref_dm=None):
    """Random subsampling simulation. ref_dm is the full-sample DM reference for MSE/coverage."""
    if ref_dm is None:
        raise ValueError("ref_dm required — run_baseline() first.")
    rng = np.random.default_rng(SEED)

    df_llm = df.dropna(subset=["f3_0", "f3_1"]).reset_index(drop=True)
    print(f"\nSubsampling pool (with LLM predictions): n={len(df_llm)}")
    print(f"  Treatment: {(df_llm.treat==1).sum()}, Control: {(df_llm.treat==0).sum()}")

    Y    = df_llm["outcome"].values
    A    = df_llm["treat"].values
    X    = df_llm[COVARIATE_COLS].values
    f3_1 = df_llm["f3_1"].values
    f3_0 = df_llm["f3_0"].values

    results = []

    for n in sample_sizes:
        n1 = n // 2; n0 = n - n1
        treat_idx   = np.where(A == 1)[0]
        control_idx = np.where(A == 0)[0]
        if len(treat_idx) < n1 or len(control_idx) < n0:
            print(f"  Skipping n={n}: not enough patients")
            continue

        sv_dm, sv_aipw, sv_haipw = [], [], []
        ate_dm, ate_aipw, ate_haipw = [], [], []
        lam1_ext_all, lam0_ext_all = [], []
        cov_dm, cov_aipw, cov_haipw = [], [], []

        for _ in range(n_reps):
            idx1 = rng.choice(treat_idx,   n1, replace=False)
            idx0 = rng.choice(control_idx, n0, replace=False)
            idx  = np.concatenate([idx1, idx0])

            Yi, Ai, Xi = Y[idx], A[idx], X[idx]
            f1i = f3_1[idx]
            f0i = f3_0[idx]
            n_sub = len(Yi)
            pi    = Ai.mean()

            # DM
            dm    = Yi[Ai==1].mean() - Yi[Ai==0].mean()
            v_dm  = (np.var(Yi[Ai==1], ddof=1) / (Ai==1).sum() +
                     np.var(Yi[Ai==0], ddof=1) / (Ai==0).sum())

            # Internal cross-fitted AIPW
            Q1_int, Q0_int = crossfit_arm_models(Yi, Ai, Xi)
            psi1_int = phi1(Yi, Ai, Q1_int, pi)
            psi0_int = phi0(Yi, Ai, Q0_int, pi)
            theta_aipw = psi1_int.mean() - psi0_int.mean()
            if_cov     = np.cov(psi1_int, psi0_int)[0, 1]
            v_aipw     = (np.var(psi1_int, ddof=1) + np.var(psi0_int, ddof=1)
                          - 2 * if_cov)

            # H-AIPW with LLM external model
            psi1_llm = phi1(Yi, Ai, f1i, pi)
            psi0_llm = phi0(Yi, Ai, f0i, pi)
            ate_h, sv_h, lam1, lam0 = haipw_ate(
                [psi1_int, psi1_llm], [psi0_int, psi0_llm])

            # Coverage
            se_dm    = np.sqrt(v_dm)
            se_aipw  = np.sqrt(v_aipw / n_sub)
            se_haipw = np.sqrt(sv_h   / n_sub)
            cov_dm.append(int(abs(dm           - ref_dm) <= 1.96 * se_dm))
            cov_aipw.append(int(abs(theta_aipw - ref_dm) <= 1.96 * se_aipw))
            cov_haipw.append(int(abs(ate_h     - ref_dm) <= 1.96 * se_haipw))

            sv_dm.append(n_sub * v_dm)
            sv_aipw.append(v_aipw)
            sv_haipw.append(sv_h)
            ate_dm.append(dm)
            ate_aipw.append(theta_aipw)
            ate_haipw.append(ate_h)
            lam1_ext_all.append(lam1[1])
            lam0_ext_all.append(lam0[1])

        mse = lambda ates: np.mean((np.array(ates) - ref_dm) ** 2)

        results.append({
            "n":                    n,
            "sv_DM":                np.mean(sv_dm),
            "sv_AIPW":              np.mean(sv_aipw),
            "sv_HAIPW_LLM":         np.mean(sv_haipw),
            "mse_DM":               mse(ate_dm),
            "mse_AIPW":             mse(ate_aipw),
            "mse_HAIPW_LLM":        mse(ate_haipw),
            "ate_DM":               np.mean(ate_dm),
            "ate_AIPW":             np.mean(ate_aipw),
            "ate_HAIPW_LLM":        np.mean(ate_haipw),
            "lam1_LLM":             round(np.mean(lam1_ext_all), 3),
            "lam0_LLM":             round(np.mean(lam0_ext_all), 3),
            "cov_DM":               round(np.mean(cov_dm),    3),
            "cov_AIPW":             round(np.mean(cov_aipw),  3),
            "cov_HAIPW_LLM":        round(np.mean(cov_haipw), 3),
        })

        print(f"  n={n:4d} | Scaled Var: DM={np.mean(sv_dm):.3f}  "
              f"AIPW={np.mean(sv_aipw):.3f}  H-AIPW(LLM)={np.mean(sv_haipw):.3f}  "
              f"Reduction={100*(np.mean(sv_aipw)-np.mean(sv_haipw))/np.mean(sv_aipw):.1f}%  "
              f"λ₁(LLM trt)={np.mean(lam1_ext_all):.3f}  λ₀(LLM ctrl)={np.mean(lam0_ext_all):.3f}")

    return pd.DataFrame(results)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_ist3()
    df = load_llm_predictions(df)

    Y_full = df["outcome"].values; A_full = df["treat"].values
    ref_dm = float(Y_full[A_full==1].mean() - Y_full[A_full==0].mean())
    print(f"\nComplete-case IST3 DM reference (OHS<=2): {ref_dm:.4f}")

    print("\nRunning LLM subsampling analysis (1000 reps per sample size)...")
    results = run_subsampling(df, ref_dm=ref_dm)

    out_path = Path(__file__).parent / "haipw_llm_results.csv"
    results.to_csv(out_path, index=False)

    print("\n" + "="*65)
    print("RESULTS SUMMARY — Arm-Specific H-AIPW with LLM")
    print("="*65)
    pct_red = 100 * (results["sv_AIPW"] - results["sv_HAIPW_LLM"]) / results["sv_AIPW"]
    display = results[["n", "sv_DM", "sv_AIPW", "sv_HAIPW_LLM",
                        "lam1_LLM", "lam0_LLM"]].copy()
    display.insert(4, "var_reduction_pct", pct_red.round(1))
    print(display.to_string(index=False))
    print(f"\nResults saved to {out_path}")
